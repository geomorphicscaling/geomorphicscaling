# ======================================================================
# fig_stream_matrix.jl
# ----------------------------------------------------------------------
# Paper figure: 3 columns x 5 rows, exported as PNG **and** PDF.
#
#   row 1     | Parameter setting | Ensemble 1 | Ensemble 2   <- headers
#   row 2     | baseline          |  render    |  render
#   rows 3-5  | multi-param scenario |  render |  render
#
# TWO STAGES (this is why the layout is reliable):
#   STAGE 1  GLMakie renders each of the 8 panels to its own PNG.
#   STAGE 2  All panels are auto-cropped to ONE shared bounding box (so the
#            scale is identical across the table), then CairoMakie composes
#            the images + text into a tight table and writes PNG + PDF.
#
# Composing *images* (fixed, known pixel sizes) instead of live Axis3 scenes
# is what makes the grid deterministic — no layout solver to fight.
#
# No window is displayed; this only writes files.
#
# RUN:
#   julia --project=. -e 'import Pkg; Pkg.instantiate()'   # once
#   julia --project=. -t auto fig_stream_matrix.jl
# ======================================================================

using GLMakie              # re-exports the Makie API (Figure, Axis3, save, …)
import CairoMakie          # second backend; activated in stage 2 for PNG + PDF
import FileIO              # imported, not `using`: FileIO also exports `save`
using NPZ, DataStructures, Colors, Printf, Statistics

# ============================ CONFIG ==================================

const GEN_RES    = 4096      # DEM generation grid (matches your analysis).
                             #   set 1024 for a fast draft; cache keys on this.
const HYDRO_RES  = 1024      # hydrology grid (DEM block-mean downsampled to this)
const MIN_ACC    = 500       # channel threshold in CELLS — tied to HYDRO_RES.
                             #   do not change either without the other.

# --- camera (matched to the reference render) ---
const AZIMUTH    = 1.50π     # looking down the y axis
const ELEVATION  = 0.14π     # ~25 deg above horizontal

# Vertical exaggeration, stated physically instead of as an opaque box ratio.
# The reference view maps 5500 m onto 50 units and 100 km onto 100 units,
# i.e. 9x. ZASPECT is derived from this below, once ZMAX_COMMON is known.
const VERT_EXAG  = 9.0

# --- panel render (stage 1) ---
const PANEL_PX_W = 760       # offscreen render size per panel (smaller => smaller PDF)
const PANEL_PX_H = 520
const PANEL_SS   = 2         # supersampling for the panel PNGs

# --- table (stage 2) ---
const PANEL_DISP_W = 620     # displayed width of each render cell, px
const TEXT_W       = 620     # width of the text column, px
const CROP_PAD     = 34      # margin kept around the terrain after auto-cropping.
                             #   the crop normalises every panel to its content, so
                             #   THIS is the "zoom out" / breathing-room control.
const HEADER_H     = 82
const COLGAP       = 10
const ROWGAP       = 8
const OUT_PNG      = "fig_stream_matrix.png"
const OUT_PDF      = "fig_stream_matrix.pdf"
const OUT_DPI      = 2       # px_per_unit for the PNG

# --- fonts (large, tabular) ---
const MONO_FONT  = Sys.iswindows() ? "Consolas" : "DejaVu Sans Mono"
const FS_HEADER  = 46
const FS_ROWNAME = 40
const FS_PARAM   = 31

# --- style ---
const STREAM_COLOR = RGBf(0.12, 0.42, 0.86)
const STREAM_W     = 2.0
const NOBASIN_COL  = RGBf(0.78, 0.78, 0.78)
const BASIN_SAT    = 0.55
const BASIN_VAL    = 0.80
const LIFT_FRAC    = 0.010
const DEPTH_SHIFT  = -0.01f0
const EPS_FILL     = 1e-4

const CACHE_DIR  = "fig_cache"
const PANEL_DIR  = "fig_panels"

const SEEDS = [7, 23]        # ensemble 1, ensemble 2

# --- the four rows ---------------------------------------------------
# Each setting perturbs a DIFFERENT mechanism, 5 parameters at a time.
# max_elev is deliberately held at baseline in all rows, so relief is
# directly comparable and the shared z box is exactly the baseline range.
#
# Note on leverage: because gain < 1, the high octaves carry little energy,
# so `octaves` and `k` are weak levers. Every setting therefore carries four
# strong parameters (gain / shaping / domain / lacunarity / radial_*) plus
# one weak one, rather than leaning on the weak pair.
const ROWS = [
    (name = "Baseline",
     changes = NamedTuple()),

    # (1) SPECTRAL ENERGY -> rough, finely dissected, high-frequency relief
    (name = "Setting 1",
     changes = (gain = 0.80, lacunarity = 2.9, shaping = 2.8, octaves = 9, k = 0.5)),

    # (2) SCALE + SMOOTHNESS -> few broad, smooth landforms
    (name = "Setting 2",
     changes = (gain = 0.30, domain = 4.5, shaping = 1.1, lacunarity = 1.6, k = 5.0)),

    # (3) RADIAL GEOMETRY -> steep radial decay to a concentrated central
    #     massif with radial drainage. The only row touching radial_*.
    (name = "Setting 3",
     changes = (radial_strength = 2.0, radial_exponent = 1.1, domain = 12.0,
                shaping = 2.3, octaves = 8)),
]

# ======================================================================
# 1) GENERATION
# ======================================================================

smoothstep5(t)  = t^3 * (t * (t * 6 - 15) + 10)
dsmoothstep5(t) = 30 * t^2 * (t - 1)^2

function hash33(ix::Int, iy::Int, iz::Int, seed::Int = 0)
    ux = reinterpret(UInt64, Int64(ix)); uy = reinterpret(UInt64, Int64(iy))
    uz = reinterpret(UInt64, Int64(iz)); us = reinterpret(UInt64, Int64(seed))
    h = ux * 0x9E3779B185EBCA87 ⊻ uy * 0xC2B2AE3D27D4EB4F ⊻
        uz * 0x165667B19E3779F9 ⊻ us * 0x85EBCA77C2B2AE63
    h ⊻= h >> 33; h *= 0xff51afd7ed558ccd
    h ⊻= h >> 33; h *= 0xc4ceb9fe1a85ec53
    h ⊻= h >> 33
    return Float64(h & 0xFFFFFFFF) / Float64(0xFFFFFFFF)
end

function value_noise_with_deriv(p::NTuple{3,Float64}; seed::Int = 0)
    x, y, z = p
    ix = floor(Int, x); iy = floor(Int, y); iz = floor(Int, z)
    fx = x - ix; fy = y - iy; fz = z - iz
    u = smoothstep5(fx);  v = smoothstep5(fy);  w = smoothstep5(fz)
    du = dsmoothstep5(fx); dv = dsmoothstep5(fy); dw = dsmoothstep5(fz)
    c000 = hash33(ix, iy, iz, seed);     c100 = hash33(ix+1, iy, iz, seed)
    c010 = hash33(ix, iy+1, iz, seed);   c110 = hash33(ix+1, iy+1, iz, seed)
    c001 = hash33(ix, iy, iz+1, seed);   c101 = hash33(ix+1, iy, iz+1, seed)
    c011 = hash33(ix, iy+1, iz+1, seed); c111 = hash33(ix+1, iy+1, iz+1, seed)
    x00 = (1-u)*c000 + u*c100; x10 = (1-u)*c010 + u*c110
    x01 = (1-u)*c001 + u*c101; x11 = (1-u)*c011 + u*c111
    y0 = (1-v)*x00 + v*x10;    y1 = (1-v)*x01 + v*x11
    val = (1-w)*y0 + w*y1
    dx00 = c100-c000; dx10 = c110-c010; dx01 = c101-c001; dx11 = c111-c011
    dy0 = x10-x00; dy1 = x11-x01; dz = y1-y0
    ddx = du * ((1-w)*((1-v)*dx00 + v*dx10) + w*((1-v)*dx01 + v*dx11))
    ddy = dv * ((1-w)*dy0 + w*dy1)
    ddz = dw * dz
    return val, ddx, ddy, ddz
end

function fbm_val(p::NTuple{3,Float64}; octaves::Int=6, lacunarity::Float64=2.0,
                 gain::Float64=0.55, k::Float64=1.0, klist=nothing, seed::Int=0)
    x, y, z = p
    total = 0.0; amp = 1.0; freq = 1.0; norm = 0.0
    for o in 1:octaves
        kval = klist === nothing ? k : klist[min(o, length(klist))]
        val, dx, dy, dz = value_noise_with_deriv((x*freq, y*freq, z*freq); seed = seed + 1013*o)
        val = 2.0*val - 1.0; dx *= 2.0; dy *= 2.0; dz *= 2.0
        g = sqrt(dx^2 + dy^2 + dz^2)
        total += amp * (1.0/(1.0 + kval*g)) * val
        norm += amp; amp *= gain; freq *= lacunarity
    end
    return total / norm
end

const BASELINE = (
    octaves = 6, lacunarity = 2.0, gain = 0.55, k = 1.0,
    warp_strength = 0.04, warp_freq = 0.6, warp_octaves = 2, warp_gain = 0.5,
    warp_lacunarity = 2.0, warp_anisotropy = 0.7,
    domain = 10.0, shaping = 1.8, radial_strength = 1.0, radial_exponent = 2.0,
    max_elev = 2500.0, z = 0.2, klist = [0.6, 0.8, 1.0, 1.4, 1.8, 2.5, 3.5],
)

const INT_PARAMS = (:octaves, :warp_octaves)
function clamp_param(name::Symbol, v)
    if name in INT_PARAMS;              return max(1, round(Int, v))
    elseif name in (:gain, :warp_gain); return clamp(float(v), 0.05, 1.0)
    elseif name in (:lacunarity, :warp_lacunarity, :domain, :warp_freq,
                    :radial_strength, :radial_exponent, :max_elev)
        return max(float(v), 1e-3)
    else;                               return max(float(v), 0.0)
    end
end
build_gp(ch) = merge(BASELINE, NamedTuple{keys(ch)}(Tuple(clamp_param(k, v) for (k, v) in pairs(ch))))

# Tallest max_elev across all rows. EVERY panel is drawn with z limits 0..ZMAX_COMMON
# so that relief is directly comparable between rows. Without this, Axis3 autoscales
# z per panel and a 1200 m terrain is stretched to look as tall as a 3500 m one.
const ZMAX_COMMON = maximum(isempty(r.changes) ? BASELINE.max_elev :
                            build_gp(r.changes).max_elev for r in ROWS)

# x spans 100 km = 100_000 m and is 1 unit wide; z spans ZMAX_COMMON over ZASPECT units.
const ZASPECT = VERT_EXAG * ZMAX_COMMON / 100_000

# Panel PNGs are cached. This tag goes in their filename so that changing the camera
# or vertical scale automatically invalidates them (the DEM cache is untouched, since
# DEMs do not depend on the view).
const STYLE_TAG = @sprintf("az%03d_el%03d_ve%02d_z%04d",
                           round(Int, AZIMUTH/π*100), round(Int, ELEVATION/π*100),
                           round(Int, VERT_EXAG), round(Int, ZMAX_COMMON))

function generate_heightmap(W::Int, H::Int; domain_size, z, octaves, lacunarity, gain, k,
                            klist, warp_strength, warp_freq, warp_octaves, warp_gain,
                            warp_lacunarity, warp_anisotropy, seed::Int = 0)
    img = Array{Float32}(undef, H, W)
    Threads.@threads for j in 1:H
        y = (j-1)/(H-1) * domain_size
        @inbounds for i in 1:W
            x = (i-1)/(W-1) * domain_size
            wv = fbm_val((x*warp_freq, y*warp_freq, z); octaves = warp_octaves,
                         lacunarity = warp_lacunarity, gain = warp_gain, k = 1.0, seed = seed)
            img[j, i] = Float32(fbm_val((x + warp_strength*wv,
                                         y + warp_strength*(warp_anisotropy*wv), z);
                octaves = octaves, lacunarity = lacunarity, gain = gain,
                k = k, klist = klist, seed = seed))
        end
    end
    return img
end

function make_one_realization(seed::Int, gp; W::Int, H::Int, min_elev::Float64 = 0.0)
    hr = generate_heightmap(W, H;
        domain_size = gp.domain, z = gp.z,
        octaves = round(Int, gp.octaves), lacunarity = gp.lacunarity,
        gain = gp.gain, k = gp.k, klist = gp.klist,
        warp_strength = gp.warp_strength, warp_freq = gp.warp_freq,
        warp_octaves = round(Int, gp.warp_octaves), warp_gain = gp.warp_gain,
        warp_lacunarity = gp.warp_lacunarity, warp_anisotropy = gp.warp_anisotropy,
        seed = seed)
    h = copy(hr)
    h .= (h .- minimum(h)) ./ (maximum(h) - minimum(h) + 1e-12)
    h .= h .^ (1 .+ gp.shaping .* h)
    cx = (W+1)/2; cy = (H+1)/2
    @inbounds for j in 1:H, i in 1:W
        rx = (i-cx)/(0.5*W); ry = (j-cy)/(0.5*H)
        h[j,i] *= exp(-(gp.radial_strength * sqrt(rx^2+ry^2)^gp.radial_exponent))
    end
    h .= (h .- minimum(h)) ./ (maximum(h) - minimum(h) + 1e-12)
    return clamp.(min_elev .+ (gp.max_elev - min_elev) .* h, min_elev, gp.max_elev)
end

# ======================================================================
# 2) HYDROLOGY
# ======================================================================

const DR = (-1,-1,-1, 0, 0, 1, 1, 1)
const DC = (-1, 0, 1,-1, 1,-1, 0, 1)

function downsample_mean(Z, cs, target)
    H, W = size(Z)
    step = max(1, round(Int, max(H, W) / target))
    step == 1 && return Float64.(Z), cs
    H2 = div(H, step); W2 = div(W, step)
    out = Array{Float64}(undef, H2, W2)
    @inbounds for j in 1:W2, i in 1:H2
        s = 0.0
        for jj in 1:step, ii in 1:step; s += Z[(i-1)*step+ii, (j-1)*step+jj]; end
        out[i, j] = s / (step*step)
    end
    return out, cs*step
end

function fill_dem(Z, eps)
    H, W = size(Z); filled = copy(Z); seen = falses(H, W)
    lin = LinearIndices((H, W)); car = CartesianIndices((H, W))
    heap = BinaryMinHeap{Tuple{Float64,Int}}()
    for j in 1:W, i in (1, H); if !seen[i,j]; seen[i,j]=true; push!(heap,(Z[i,j],lin[i,j])); end; end
    for i in 1:H, j in (1, W); if !seen[i,j]; seen[i,j]=true; push!(heap,(Z[i,j],lin[i,j])); end; end
    while !isempty(heap)
        e, l = pop!(heap); c = car[l]; i = c[1]; j = c[2]
        @inbounds for k in 1:8
            ni = i+DR[k]; nj = j+DC[k]
            (1 <= ni <= H && 1 <= nj <= W) || continue
            seen[ni,nj] && continue
            seen[ni,nj] = true
            nh = max(Z[ni,nj], e + eps); filled[ni,nj] = nh
            push!(heap, (nh, lin[ni,nj]))
        end
    end
    return filled
end

function flow_accum_d8(filled, cs)
    H, W = size(filled); lin = LinearIndices((H, W))
    recv = zeros(Int, H*W)
    dist = (sqrt(2)*cs, cs, sqrt(2)*cs, cs, cs, sqrt(2)*cs, cs, sqrt(2)*cs)
    @inbounds for j in 1:W, i in 1:H
        fc = filled[i,j]; best = 0; bs = 0.0
        for k in 1:8
            ni = i+DR[k]; nj = j+DC[k]
            (1 <= ni <= H && 1 <= nj <= W) || continue
            drop = fc - filled[ni,nj]
            if drop > 0
                s = drop/dist[k]
                if s > bs; bs = s; best = lin[ni,nj]; end
            end
        end
        recv[lin[i,j]] = best
    end
    order = sortperm(vec(filled); rev = true)
    acc = ones(Float64, H*W)
    @inbounds for l in order; r = recv[l]; r != 0 && (acc[r] += acc[l]); end
    return recv, acc
end

function terminal_labels(recv)
    N = length(recv); term = zeros(Int, N)
    for s in 1:N
        term[s] != 0 && continue
        l = s; path = Int[]
        while recv[l] != 0 && term[l] == 0; push!(path, l); l = recv[l]; end
        t = term[l] != 0 ? term[l] : l
        for q in path; term[q] = t; end
        term[s] == 0 && (term[s] = t)
    end
    return term
end

# ======================================================================
# 3) CACHE + STAGE 1: render each panel to its own PNG
# ======================================================================

function get_dem(rowidx::Int, seed::Int, gp)
    isdir(CACHE_DIR) || mkpath(CACHE_DIR)
    f = joinpath(CACHE_DIR, @sprintf("row%d_seed%03d_%d.npz", rowidx, seed, GEN_RES))
    if isfile(f)
        println(@sprintf("    • DEM cached      row %d seed %03d", rowidx, seed))
        return Float64.(npzread(f)["height_m"])
    end
    t = @elapsed Z = make_one_realization(seed, gp; W = GEN_RES, H = GEN_RES)
    npzwrite(f; height_m = Float32.(Z))
    println(@sprintf("    ✅ DEM generated  row %d seed %03d  (%.1f s)", rowidx, seed, t))
    return Float64.(Z)
end

function render_panel(rowidx::Int, seed::Int, gp)
    isdir(PANEL_DIR) || mkpath(PANEL_DIR)
    out = joinpath(PANEL_DIR, @sprintf("panel_row%d_seed%03d_%d_%s.png",
                                       rowidx, seed, GEN_RES, STYLE_TAG))
    if isfile(out)
        println(@sprintf("    • panel cached    row %d seed %03d", rowidx, seed))
        return out
    end

    Zfull = get_dem(rowidx, seed, gp)
    cs_full = 100_000.0 / (GEN_RES - 1)
    Z, cs = downsample_mean(Zfull, cs_full, HYDRO_RES)
    H, W = size(Z); N = H*W
    car = CartesianIndices((H, W))

    filled    = fill_dem(Z, EPS_FILL)
    recv, acc = flow_accum_d8(filled, cs)
    ischannel = falses(N)
    @inbounds for l in 1:N
        (acc[l] >= MIN_ACC && recv[l] != 0) && (ischannel[l] = true)
    end
    chan = findall(ischannel)

    zmin, zmax = extrema(Z)
    lift = LIFT_FRAC * (zmax - zmin)
    xs = collect(0:W-1) .* cs ./ 1000.0
    ys = collect(0:H-1) .* cs ./ 1000.0
    pt(l) = (c = car[l]; Point3f(xs[c[2]], ys[c[1]], Z[c[1], c[2]] + lift))

    term = terminal_labels(recv)
    counts = Dict{Int,Int}(); for t in term; counts[t] = get(counts, t, 0) + 1; end
    haschan = Set{Int}(); for l in chan; push!(haschan, term[l]); end
    ids = sort(collect(haschan); by = k -> counts[k], rev = true)
    colmap = Dict{Int,RGBf}()
    for (n, id) in enumerate(ids)
        colmap[id] = RGBf(HSV(mod(137.507*(n-1), 360.0), BASIN_SAT, BASIN_VAL))
    end
    C = Array{RGBf}(undef, H, W)
    @inbounds for l in 1:N; C[car[l]] = get(colmap, term[l], NOBASIN_COL); end

    ndon = zeros(Int, N)
    for l in chan; r = recv[l]; (r != 0 && ischannel[r]) && (ndon[r] += 1); end
    poly = Point3f[]; NANP = Point3f(NaN, NaN, NaN)
    for s in chan
        (ndon[s] == 0 || ndon[s] >= 2) || continue
        pts = Point3f[pt(s)]; l = s
        while true
            r = recv[l]; (r == 0 || !ischannel[r]) && break
            push!(pts, pt(r)); l = r
            ndon[l] >= 2 && break
        end
        if length(pts) >= 2
            isempty(poly) || push!(poly, NANP)
            append!(poly, pts)
        end
    end

    GLMakie.activate!()
    fig = Figure(size = (PANEL_PX_W, PANEL_PX_H), backgroundcolor = :white, figure_padding = 0)
    ax = Axis3(fig[1, 1]; aspect = (1, 1, ZASPECT), azimuth = AZIMUTH,
               elevation = ELEVATION,
               viewmode = :fit,                      # :fit keeps the WHOLE terrain in frame
               protrusions = 0,                      # (:fitzoom crops the corners)
               limits = (0, 100, 0, 100, 0, ZMAX_COMMON))   # shared z => comparable relief
    surface!(ax, xs, ys, permutedims(Z); color = permutedims(C))   # default shading = relief
    isempty(poly) || lines!(ax, poly; color = STREAM_COLOR, linewidth = STREAM_W,
                            depth_shift = DEPTH_SHIFT)
    hidedecorations!(ax); hidespines!(ax)
    save(out, fig; px_per_unit = PANEL_SS)
    println(@sprintf("    ✅ panel rendered row %d seed %03d  (%d basins, %d channel cells)",
                     rowidx, seed, length(ids), length(chan)))
    return out
end

# ======================================================================
# 4) CROP: one shared bounding box for every panel
# ======================================================================

is_bg(px) = (red(px) > 0.985 && green(px) > 0.985 && blue(px) > 0.985)

function content_bbox(img)
    H, W = size(img)
    r0, r1, c0, c1 = H, 1, W, 1
    @inbounds for i in 1:H, j in 1:W
        if !is_bg(img[i, j])
            i < r0 && (r0 = i); i > r1 && (r1 = i)
            j < c0 && (c0 = j); j > c1 && (c1 = j)
        end
    end
    r0 > r1 && return (1, H, 1, W)
    return (r0, r1, c0, c1)
end

# ======================================================================
# 5) MAIN
# ======================================================================

fmtval(k, v) = k in INT_PARAMS ? string(round(Int, v)) :
               k === :max_elev  ? @sprintf("%.0f m", v) : @sprintf("%.2f", v)

const SHOWN = [:octaves, :lacunarity, :gain, :k, :domain, :shaping,
               :radial_strength, :radial_exponent, :max_elev]

function row_text(row)
    if isempty(row.changes)
        return join([@sprintf("%-16s = %s", string(k), fmtval(k, getfield(BASELINE, k)))
                     for k in SHOWN], "\n")
    end
    gp = build_gp(row.changes)
    lines = [@sprintf("%-16s %s → %s", string(k)*":",
                      fmtval(k, getfield(BASELINE, k)), fmtval(k, getfield(gp, k)))
             for (k, v) in pairs(row.changes)]
    push!(lines, ""); push!(lines, "all others at baseline")
    return join(lines, "\n")
end

function main()
    nrow = length(ROWS)
    println("="^62)
    println("STAGE 1 — rendering $(nrow*length(SEEDS)) panels")
    println("  GEN_RES=$GEN_RES  HYDRO_RES=$HYDRO_RES  MIN_ACC=$MIN_ACC  threads=$(Threads.nthreads())")
    @printf("  camera: azimuth=%.2fπ  elevation=%.2fπ  viewmode=:fit\n", AZIMUTH/π, ELEVATION/π)
    @printf("  z axis: 0–%.0f m on every panel (shared) | vertical exaggeration %.1f× | zaspect=%.3f\n",
            ZMAX_COMMON, VERT_EXAG, ZASPECT)
    println("="^62)

    paths = Matrix{String}(undef, nrow, length(SEEDS))
    for (i, row) in enumerate(ROWS)
        gp = isempty(row.changes) ? BASELINE : build_gp(row.changes)
        println("  row $i: $(row.name)")
        for (j, seed) in enumerate(SEEDS)
            paths[i, j] = render_panel(i, seed, gp)
        end
    end

    println("\n" * "="^62)
    println("STAGE 2 — cropping to a shared box and composing the table")
    println("="^62)

    imgs = [FileIO.load(paths[i, j]) for i in 1:nrow, j in 1:length(SEEDS)]
    boxes = [content_bbox(im) for im in imgs]
    r0 = minimum(b[1] for b in boxes); r1 = maximum(b[2] for b in boxes)
    c0 = minimum(b[3] for b in boxes); c1 = maximum(b[4] for b in boxes)
    pad = CROP_PAD
    r0 = max(1, r0-pad); c0 = max(1, c0-pad)
    r1 = min(size(imgs[1], 1), r1+pad); c1 = min(size(imgs[1], 2), c1+pad)
    cropped = [im[r0:r1, c0:c1] for im in imgs]
    ph_px, pw_px = size(cropped[1])
    @printf("  shared crop box: %d x %d px  (rows %d:%d, cols %d:%d)\n", pw_px, ph_px, r0, r1, c0, c1)

    disp_h = round(Int, PANEL_DISP_W * ph_px / pw_px)
    figw = TEXT_W + 2*PANEL_DISP_W + 3*COLGAP
    figh = HEADER_H + nrow*disp_h + (nrow)*ROWGAP
    @printf("  figure: %d x %d pt  ->  PNG %d x %d px\n", figw, figh, figw*OUT_DPI, figh*OUT_DPI)

    CairoMakie.activate!()
    fig = Figure(size = (figw, figh), backgroundcolor = :white, figure_padding = 12)

    for (j, txt) in enumerate(["Parameter setting", "Ensemble 1", "Ensemble 2"])
        Label(fig[1, j], txt; fontsize = FS_HEADER, font = :bold, color = :black,
              halign = j == 1 ? :left : :center, tellwidth = false, tellheight = false)
    end

    for (i, row) in enumerate(ROWS)
        r = i + 1
        box = GridLayout(fig[r, 1]; halign = :left, valign = :center, tellwidth = false)
        Label(box[1, 1], row.name; fontsize = FS_ROWNAME, font = :bold, color = :black,
              halign = :left, tellwidth = false)
        Label(box[2, 1], row_text(row); fontsize = FS_PARAM, font = MONO_FONT,
              color = :black, halign = :left, justification = :left, tellwidth = false)
        rowgap!(box, 12)

        for j in 1:length(SEEDS)
            ax = Axis(fig[r, j+1]; aspect = DataAspect(), backgroundcolor = :white)
            image!(ax, rotr90(cropped[i, j]))
            hidedecorations!(ax); hidespines!(ax)
        end
    end

    colsize!(fig.layout, 1, Fixed(TEXT_W))
    colsize!(fig.layout, 2, Fixed(PANEL_DISP_W))
    colsize!(fig.layout, 3, Fixed(PANEL_DISP_W))
    rowsize!(fig.layout, 1, Fixed(HEADER_H))
    for r in 2:(nrow+1); rowsize!(fig.layout, r, Fixed(disp_h)); end
    colgap!(fig.layout, COLGAP)
    rowgap!(fig.layout, ROWGAP)

    save(OUT_PNG, fig; px_per_unit = OUT_DPI)
    println("  ✅ $OUT_PNG")
    save(OUT_PDF, fig)
    println("  ✅ $OUT_PDF  (vector text + embedded raster panels)")
    println("\nDone. Delete $PANEL_DIR/ to force re-render; $CACHE_DIR/ to regenerate DEMs.")
    return nothing
end

main()
