# ======================================================================
# terrain_slider_app.jl
# ----------------------------------------------------------------------
# Interactive, native GLMakie app: drag sliders -> live 3D terrain.
#
# Sliders expose the 9 parameters you sweep in the notebook, plus a seed.
# Live preview is computed on a small grid (PREVIEW_RES) on every change.
# An "Export 4096²" button regenerates at full resolution and writes both
# an .npz (matching your npzwrite schema, with the 9 params embedded) and
# a styled PNG render.
#
# Generation pipeline is copied verbatim from the parameterized notebook
# (value_noise_with_deriv / fbm_with_deriv / generate_heightmap /
#  make_one_realization), so live previews match your full-res output
# exactly at equal grid size.
#
# RUN (threads make live updates snappier):
#   julia -t auto terrain_slider_app.jl
# or from a REPL:
#   include("terrain_slider_app.jl")
# ======================================================================

using GLMakie
using NPZ
using Printf
using Random
using Dates

GLMakie.activate!()
# Reached through GLMakie so we don't need Makie as a direct dependency.
const throttle = GLMakie.Makie.Observables.throttle

# ======================================================================
# 1) NOISE + FBM (verbatim from the notebook)
# ======================================================================

smoothstep5(t)  = t^3 * (t * (t * 6 - 15) + 10)
dsmoothstep5(t) = 30 * t^2 * (t - 1)^2

function hash33(ix::Int, iy::Int, iz::Int, seed::Int = 0)
    ux = reinterpret(UInt64, Int64(ix))
    uy = reinterpret(UInt64, Int64(iy))
    uz = reinterpret(UInt64, Int64(iz))
    us = reinterpret(UInt64, Int64(seed))

    h = ux * 0x9E3779B185EBCA87 ⊻
        uy * 0xC2B2AE3D27D4EB4F ⊻
        uz * 0x165667B19E3779F9 ⊻
        us * 0x85EBCA77C2B2AE63

    h ⊻= h >> 33
    h *= 0xff51afd7ed558ccd
    h ⊻= h >> 33
    h *= 0xc4ceb9fe1a85ec53
    h ⊻= h >> 33

    return Float64(h & 0xFFFFFFFF) / Float64(0xFFFFFFFF)
end

function value_noise_with_deriv(p::NTuple{3, Float64}; seed::Int = 0)
    x, y, z = p

    ix = floor(Int, x); iy = floor(Int, y); iz = floor(Int, z)
    fx = x - ix;        fy = y - iy;        fz = z - iz

    u = smoothstep5(fx);  v = smoothstep5(fy);  w = smoothstep5(fz)
    du = dsmoothstep5(fx); dv = dsmoothstep5(fy); dw = dsmoothstep5(fz)

    c000 = hash33(ix,     iy,     iz,     seed)
    c100 = hash33(ix + 1, iy,     iz,     seed)
    c010 = hash33(ix,     iy + 1, iz,     seed)
    c110 = hash33(ix + 1, iy + 1, iz,     seed)
    c001 = hash33(ix,     iy,     iz + 1, seed)
    c101 = hash33(ix + 1, iy,     iz + 1, seed)
    c011 = hash33(ix,     iy + 1, iz + 1, seed)
    c111 = hash33(ix + 1, iy + 1, iz + 1, seed)

    x00 = (1 - u) * c000 + u * c100
    x10 = (1 - u) * c010 + u * c110
    x01 = (1 - u) * c001 + u * c101
    x11 = (1 - u) * c011 + u * c111

    y0 = (1 - v) * x00 + v * x10
    y1 = (1 - v) * x01 + v * x11

    val = (1 - w) * y0 + w * y1

    dx00 = c100 - c000; dx10 = c110 - c010
    dx01 = c101 - c001; dx11 = c111 - c011
    dy0  = x10 - x00;   dy1  = x11 - x01
    dz   = y1 - y0

    dval_dx = du * ((1 - w) * ((1 - v) * dx00 + v * dx10) +
                    w       * ((1 - v) * dx01 + v * dx11))
    dval_dy = dv * ((1 - w) * dy0 + w * dy1)
    dval_dz = dw * dz

    return val, dval_dx, dval_dy, dval_dz
end

function fbm_with_deriv(
    p::NTuple{3, Float64};
    octaves::Int = 6,
    lacunarity::Float64 = 2.0,
    gain::Float64 = 0.55,
    k::Float64 = 1.0,
    klist = nothing,
    seed::Int = 0,
)
    x, y, z = p

    total = 0.0
    dtotal_dx = 0.0; dtotal_dy = 0.0; dtotal_dz = 0.0
    amp = 1.0; freq = 1.0; norm = 0.0

    for o in 1:octaves
        kval = klist === nothing ? k : klist[min(o, length(klist))]

        val, dx, dy, dz = value_noise_with_deriv(
            (x * freq, y * freq, z * freq);
            seed = seed + 1013 * o,
        )

        val = 2.0 * val - 1.0
        dx *= 2.0; dy *= 2.0; dz *= 2.0

        gradmag = sqrt(dx^2 + dy^2 + dz^2)
        weight = 1.0 / (1.0 + kval * gradmag)

        total     += amp * weight * val
        dtotal_dx += amp * weight * dx * freq
        dtotal_dy += amp * weight * dy * freq
        dtotal_dz += amp * weight * dz * freq

        norm += amp
        amp  *= gain
        freq *= lacunarity
    end

    return total / norm, dtotal_dx / norm, dtotal_dy / norm, dtotal_dz / norm
end

# ======================================================================
# 2) HEIGHTMAP + REALIZATION (verbatim from the notebook)
# ======================================================================

function generate_heightmap(
    W::Int, H::Int;
    domain_size::Float64, z::Float64,
    octaves::Int, lacunarity::Float64, gain::Float64, k::Float64,
    klist::Vector{Float64},
    warp_strength::Float64, warp_freq::Float64, warp_octaves::Int,
    warp_gain::Float64, warp_lacunarity::Float64, warp_anisotropy::Float64,
    seed::Int = 0,
)
    img = Array{Float32}(undef, H, W)

    Threads.@threads for j in 1:H
        y = (j - 1) / (H - 1) * domain_size
        @inbounds for i in 1:W
            x = (i - 1) / (W - 1) * domain_size

            warp_val, _, _, _ = fbm_with_deriv(
                (x * warp_freq, y * warp_freq, z);
                octaves = warp_octaves, lacunarity = warp_lacunarity,
                gain = warp_gain, k = 1.0, seed = seed,
            )

            xw = x + warp_strength * warp_val
            yw = y + warp_strength * (warp_anisotropy * warp_val)

            Hval, _, _, _ = fbm_with_deriv(
                (xw, yw, z);
                octaves = octaves, lacunarity = lacunarity,
                gain = gain, k = k, klist = klist, seed = seed,
            )

            img[j, i] = Float32(Hval)
        end
    end

    return img
end

# Baseline parameter set (non-swept knobs stay at these values)
const BASELINE = (
    octaves         = 6,
    lacunarity      = 2.0,
    gain            = 0.55,
    k               = 1.0,
    warp_strength   = 0.04,
    warp_freq       = 0.6,
    warp_octaves    = 2,
    warp_gain       = 0.5,
    warp_lacunarity = 2.0,
    warp_anisotropy = 0.7,
    domain          = 10.0,
    shaping         = 1.8,
    radial_strength = 1.0,
    radial_exponent = 2.0,
    max_elev        = 2500.0,
    z               = 0.2,
    klist           = [0.6, 0.8, 1.0, 1.4, 1.8, 2.5, 3.5],
)

const INT_PARAMS = (:octaves, :warp_octaves)

function clamp_param(name::Symbol, v)
    if name in INT_PARAMS
        return max(1, round(Int, v))
    elseif name in (:gain, :warp_gain)
        return clamp(float(v), 0.05, 0.95)
    elseif name in (:lacunarity, :warp_lacunarity, :domain, :warp_freq,
                    :radial_strength, :radial_exponent, :max_elev)
        return max(float(v), 1e-3)
    else  # k, warp_strength, warp_anisotropy, shaping
        return max(float(v), 0.0)
    end
end

function make_one_realization(
    seed::Int, gp = BASELINE;
    W::Int = 4096, H::Int = 4096, min_elev::Float64 = 0.0,
)
    height_raw = generate_heightmap(
        W, H;
        domain_size = gp.domain, z = gp.z,
        octaves = round(Int, gp.octaves), lacunarity = gp.lacunarity,
        gain = gp.gain, k = gp.k, klist = gp.klist,
        warp_strength = gp.warp_strength, warp_freq = gp.warp_freq,
        warp_octaves = round(Int, gp.warp_octaves), warp_gain = gp.warp_gain,
        warp_lacunarity = gp.warp_lacunarity, warp_anisotropy = gp.warp_anisotropy,
        seed = seed,
    )

    height = copy(height_raw)
    height .= (height .- minimum(height)) ./ (maximum(height) - minimum(height) + 1e-12)
    height .= height .^ (1 .+ gp.shaping .* height)

    cx = (W + 1) / 2; cy = (H + 1) / 2
    @inbounds for j in 1:H, i in 1:W
        rx = (i - cx) / (0.5 * W)
        ry = (j - cy) / (0.5 * H)
        r = sqrt(rx^2 + ry^2)
        height[j, i] *= exp(-(gp.radial_strength * r^gp.radial_exponent))
    end

    height .= (height .- minimum(height)) ./ (maximum(height) - minimum(height) + 1e-12)

    height_m = clamp.(min_elev .+ (gp.max_elev - min_elev) .* height, min_elev, gp.max_elev)
    @assert all(isfinite, height_m)
    return height_m
end

# ======================================================================
# 3) CONFIG
# ======================================================================

const PREVIEW_RES   = 256          # live preview grid (square). 192 = snappier, 384 = sharper
const EXPORT_RES    = 4096         # full-res export grid
const COVERAGE_KM   = 100.0
const MIN_ELEV      = 0.0
const EXPORT_DIR    = "live_exports"

# Render style (matches your terrain_render)
const DOWNSAMPLE    = 2
const AZIMUTH       = 1.275π
const ELEVATION     = 0.15π
const ZASPECT       = 0.50
const COLORMAP      = :terrain
const LABELSIZE     = 22
const TICKLABELSIZE = 16

# Slider specs: (symbol, label, lo, hi, baseline, integer?)
# lo/hi come from baseline ± your param_dx windows.
const SLIDER_SPECS = [
    (:octaves,         "octaves",          1.0,    10.0,    6.0,    true),
    (:lacunarity,      "lacunarity",       0.5,    5.5,    2.0,    false),
    (:gain,            "gain",             0.10,   1.0,   0.55,   false),
    (:k,               "k",                0.5,    7.5,    1.0,    false),
    (:domain,          "domain",           0.1,    15.0,   10.0,   false),
    (:shaping,         "shaping",          0.1,    5.0,    1.8,    false),
    (:radial_strength, "radial_strength",  0.1,    2.5,    1.0,    false),
    (:radial_exponent, "radial_exponent",  0.1,    7.5,    2.0,    false),
    (:max_elev,        "max_elev (m)",     500.0, 5500.0, 2500.0, false),
]
const SEED_RANGE = 1:1:100
const SEED_START = 1

# Stable vertical box = highest possible max_elev, so the terrain doesn't
# rescale as you drag.
const MAX_ELEV_CEIL = maximum(s[4] for s in SLIDER_SPECS if s[1] === :max_elev)

const UPDATE_INTERVAL = 0.08       # seconds between live recomputes while dragging

# ======================================================================
# 4) HELPERS
# ======================================================================

# Build a full parameter NamedTuple from the 9 slider values (clamped exactly
# like the notebook). Order matches SLIDER_SPECS.
function build_gp(vals)
    overrides = NamedTuple{Tuple(s[1] for s in SLIDER_SPECS)}(
        Tuple(clamp_param(SLIDER_SPECS[i][1], vals[i]) for i in 1:length(SLIDER_SPECS))
    )
    return merge(BASELINE, overrides)
end

# Compute a preview heightmap, oriented for surface!(xs, ys, Z).
function compute_preview(vals, seed)
    gp = build_gp(vals)
    Z = make_one_realization(round(Int, seed), gp;
                             W = PREVIEW_RES, H = PREVIEW_RES, min_elev = MIN_ELEV)
    return permutedims(Z)
end

# Axis coordinates in km for a given grid resolution (extent fixed to COVERAGE_KM).
function axis_km(res::Int)
    cs = COVERAGE_KM * 1000.0 / (res - 1)   # meters per cell
    xs = collect(0:res-1) .* cs ./ 1000.0
    return xs, cs
end

# Offscreen styled PNG render of a full height matrix (mirrors terrain_render).
function render_terrain_png(Z::AbstractMatrix, cellsize_m::Float64, path::String)
    ny, nx = size(Z)
    ii = 1:DOWNSAMPLE:nx
    jj = 1:DOWNSAMPLE:ny
    Zt = permutedims(Z[jj, ii])
    xs = collect(ii .- 1) .* cellsize_m ./ 1000.0
    ys = collect(jj .- 1) .* cellsize_m ./ 1000.0

    fig = Figure(size = (1450, 1000), figure_padding = 12, backgroundcolor = :white)
    ax = Axis3(fig[1, 1];
        aspect = (1, 1, ZASPECT), azimuth = AZIMUTH, elevation = ELEVATION,
        viewmode = :fitzoom,
        xlabel = "x (km)", ylabel = "y (km)", zlabel = "elevation (m)",
        xlabelsize = 44, ylabelsize = 44, zlabelsize = 44,
        xticklabelsize = 30, yticklabelsize = 30, zticklabelsize = 30,
        xlabeloffset = 42, ylabeloffset = 70, zlabeloffset = 90,
        xgridvisible = false, ygridvisible = false, zgridvisible = false,
        protrusions = (70, 85, 60, 75),
    )
    surface!(ax, xs, ys, Zt; colormap = COLORMAP)
    save(path, fig; px_per_unit = 2)
    return path
end

# ======================================================================
# 5) BUILD THE APP
# ======================================================================

xs_prev, _ = axis_km(PREVIEW_RES)
ys_prev = xs_prev

fig = Figure(size = (1500, 950), backgroundcolor = :white)

# --- 3D terrain (left) ---
ax = Axis3(fig[1, 1];
    aspect = (1, 1, ZASPECT), azimuth = AZIMUTH, elevation = ELEVATION,
    viewmode = :fitzoom,
    limits = (0, COVERAGE_KM, 0, COVERAGE_KM, MIN_ELEV, MAX_ELEV_CEIL),
    xlabel = "x (km)", ylabel = "y (km)", zlabel = "elevation (m)",
    xlabelsize = LABELSIZE, ylabelsize = LABELSIZE, zlabelsize = LABELSIZE,
    xticklabelsize = TICKLABELSIZE, yticklabelsize = TICKLABELSIZE,
    zticklabelsize = TICKLABELSIZE,
    xgridvisible = false, ygridvisible = false, zgridvisible = false,
)

# --- control panel (right) ---
right = fig[1, 2] = GridLayout()

slider_tuples = [
    (label = s[2],
     range = s[6] ? (round(Int, s[3]):1:round(Int, s[4])) : range(s[3], s[4], length = 161),
     startvalue = s[6] ? round(Int, s[5]) : s[5],
     format = s[6] ? (x -> string(round(Int, x))) : (x -> @sprintf("%.3f", x)))
    for s in SLIDER_SPECS
]
push!(slider_tuples,
    (label = "seed", range = SEED_RANGE, startvalue = SEED_START,
     format = x -> string(round(Int, x))))

sg = SliderGrid(right[1, 1], slider_tuples...; tellheight = false)

param_sliders = sg.sliders[1:length(SLIDER_SPECS)]
seed_slider   = sg.sliders[end]

current_vals() = [s.value[] for s in param_sliders]
current_seed() = seed_slider.value[]

# --- buttons + status ---
btnrow = right[2, 1] = GridLayout()
export_btn = Button(btnrow[1, 1]; label = "Export $(EXPORT_RES)² (.npz + PNG)")
reset_btn  = Button(btnrow[1, 2]; label = "Reset to baseline")

status = Observable("Ready. Drag a slider to update the terrain.")
Label(right[3, 1], status; tellwidth = false, halign = :left, fontsize = 15)

colsize!(fig.layout, 1, Relative(0.66))
rowgap!(right, 12)

# --- live surface ---
Z_obs = Observable(compute_preview(current_vals(), current_seed()))
surface!(ax, xs_prev, ys_prev, Z_obs;
         colormap = COLORMAP, colorrange = (MIN_ELEV, MAX_ELEV_CEIL))

# Combine all slider values into one observable, throttle, recompute.
combined = lift((vals...) -> collect(vals),
                (s.value for s in param_sliders)..., seed_slider.value)

on(throttle(UPDATE_INTERVAL, combined)) do v
    vals = v[1:end-1]
    seed = v[end]
    Z_obs[] = compute_preview(vals, seed)
end

# --- reset ---
on(reset_btn.clicks) do _
    for (s, spec) in zip(param_sliders, SLIDER_SPECS)
        set_close_to!(s, spec[6] ? round(Int, spec[5]) : spec[5])
    end
    set_close_to!(seed_slider, SEED_START)
    status[] = "Reset to baseline."
end

# --- export full-res ---
on(export_btn.clicks) do _
    vals = current_vals(); seed = round(Int, current_seed())
    status[] = "Exporting at $(EXPORT_RES)²… the window may freeze a few seconds."
    yield()

    gp = build_gp(vals)
    Zf = make_one_realization(seed, gp; W = EXPORT_RES, H = EXPORT_RES, min_elev = MIN_ELEV)
    cellsize_m = COVERAGE_KM * 1000.0 / (EXPORT_RES - 1)
    zmin, zmax = extrema(Zf)

    mkpath(EXPORT_DIR)
    stamp   = Dates.format(now(), "yyyymmdd_HHMMSS")
    base    = joinpath(EXPORT_DIR, @sprintf("terrain_seed%03d_%s", seed, stamp))
    outnpz  = base * ".npz"
    outpng  = base * ".png"

    npzwrite(outnpz;
        height_m        = Float32.(Zf),
        seed            = Int32[seed],
        W               = Int32[EXPORT_RES], H = Int32[EXPORT_RES],
        coverage_km     = Float32[COVERAGE_KM],
        cellsize_m      = Float64[cellsize_m],
        min_elev        = Float32[MIN_ELEV],
        max_elev        = Float32[gp.max_elev],
        # embed the 9 swept params so each export is self-describing
        octaves         = Int32[round(Int, gp.octaves)],
        lacunarity      = Float64[gp.lacunarity],
        gain            = Float64[gp.gain],
        k               = Float64[gp.k],
        domain          = Float64[gp.domain],
        shaping         = Float64[gp.shaping],
        radial_strength = Float64[gp.radial_strength],
        radial_exponent = Float64[gp.radial_exponent],
    )

    render_terrain_png(Zf, cellsize_m, outpng)

    status[] = @sprintf("Saved → %s  (range %.0f–%.0f m) and matching .png", outnpz, zmin, zmax)
    println(status[])
end

# ======================================================================
# 6) SHOW
# ======================================================================

println("Launching terrain slider app…")
println("  preview grid : $(PREVIEW_RES)×$(PREVIEW_RES)")
println("  export grid  : $(EXPORT_RES)×$(EXPORT_RES)")
println("  threads      : $(Threads.nthreads())  (run with `julia -t auto` for snappier live updates)")

screen = display(fig)
isinteractive() || wait(screen)   # keep window open when run as a script
