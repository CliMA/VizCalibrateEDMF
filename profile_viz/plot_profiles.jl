using NCDatasets
const NC = NCDatasets
# using GLMakie
using CairoMakie
using Glob
import Dierckx
import Statistics
import ColorSchemes
include("helper_funcs.jl")


colorscheme = ColorSchemes.Paired_10

# configs
sim_configs = Dict(
    # "stoch.calibrated" => "/Users/haakon/Documents/CliMA/CEDMF_output/output/rico_stoch_calibrated",
    "deterministic" => "/groups/esm/hervik/calibration/output/eki_rico_trmm/LES/results_Inversion_dt_1.0_p16_e100_i40_d19_LES_2022-05-10_19-51_HIA_ref/fwd_maps",
    "LES" => "/groups/esm/hervik/calibration/static_input/anna_LES_data_220531",
)
N_configs = length(sim_configs)

ref_SCM_config = "deterministic"  # One of the SCM `sim_config` keys that can be used as reference for various tasks
case_name = "Rico"  # name of the case to be plotted, i.e. related to `.../data/Output.<case_name>*/stats/Stats.<case_name>.nc`

# Averaging times
t_start = 22.0 * 3600
t_stop = 24.0 * 3600

# Variables to plot
profile_vars = [
    "ql_mean", "Hvar_mean", "QTvar_mean", "qi_mean", "thetal_mean", 
    "v_mean", "updraft_qt", "qt_mean", "updraft_area", "updraft_w", 
    "temperature_mean", "updraft_thetal", "u_mean", "tke_mean", "nondim_entrainment_sc",
    "nondim_detrainment_sc", "entrainment_sc", "detrainment_sc", "qr_mean",
    
]

# z interpolation levels
z_f_interp = Dict{String, Vector{Float64}}()
z_c_interp = Dict{String, Vector{Float64}}()
is_face_var = Dict{String, Bool}()
for (sim_label, sim_path) in sim_configs
    ncfile_z_interp = first(glob.("Output.$case_name*/stats/Stats.$case_name.nc", sim_path))
    NC.Dataset(ncfile_z_interp) do ds
        z_f_interp[sim_label] = get_nc_data(ds, "zf")
        z_c_interp[sim_label] = get_nc_data(ds, "zc")
        if sim_label == ref_SCM_config
            merge!(is_face_var, Dict(profile_vars .=> is_face_variable.(Ref(ds), profile_vars)))
        end
    end
end

z_max = maximum(z_f_interp[ref_SCM_config])

# iterate simulation configurations
all_profiles = Dict{String, Any}()

for (i, (sim_label, sim_path)) in enumerate(sim_configs)
    @info "computing profiles for $sim_label"
    # get ensemble data for the group
    local nc_files = glob("Output.$case_name*/stats/Stats.$case_name.nc", sim_path)
    profiles = all_profiles[sim_label] = Dict()
    # fetch time-averaged statistics for each ensemble member
    for nc_file in nc_files
        NC.Dataset(nc_file) do ds  # open dataset
            # get time and space coords
            time_arr = get_time(ds, "t")
            z_c = get_nc_data(ds, "zc")[:]
            z_f = get_nc_data(ds, "zf")[:]
            # Find the nearest matching final time:
            t_cmp = min(time_arr[end], t_stop)
            @info "time compared: $t_cmp"
            if t_cmp ≤ t_start
                throw(ArgumentError("t_end has to be after t_start for simulation: $ncfile"))
            end

            # iterate profiles
            for tc_var in profile_vars
                # @show tc_var
                # fetch data
                data_ds_arr = get_nc_data(ds, tc_var)

                z = is_face_var[tc_var] ? z_f : z_c
                z_interp = is_face_var[tc_var] ? z_f_interp[sim_label] : z_c_interp[sim_label]
                # compute time-averaged profile
                data_arr = if isnothing(data_ds_arr)
                    fill(NaN, length(time_arr), length(z))
                else
                    data_ds_arr[:]'
                end
                data_cont = Dierckx.Spline2D(time_arr, z, data_arr; kx = 1, ky = 1)  # interpolate
                R = range(t_start, t_cmp; length = 50)
                data_cont_mapped = map(z_interp) do zi
                    Statistics.mean(map(t -> data_cont(t, zi), R))
                end
                
                # Append data to matrix of data (rows = z, cols = ens_i)
                if haskey(profiles, tc_var)
                    profiles[tc_var] = hcat(profiles[tc_var], data_cont_mapped)
                else
                    profiles[tc_var] = data_cont_mapped
                end
            end  # end profile_vars iter
        end  # close ds
    end  # end nc_files iter

end  # end ens_path iter

# Make plots
rows = 4; cols = 5
fig = Figure(resolution = (500cols, 500rows))
layout_inds = vec(permutedims(CartesianIndices((rows,cols))))
for (ind, tc_var) in zip(layout_inds, profile_vars)
    local ax = Axis(fig[ind.I...], title = tc_var)

    for (i, (sim_label, sim_path)) in enumerate(sim_configs)
        local color = ColorSchemes.get(colorscheme, i / N_configs)

        local z = is_face_var[tc_var] ? z_f_interp[sim_label] : z_c_interp[sim_label]
        local z_km = z ./ 1e3

        local data = all_profiles[sim_label][tc_var]
        local data_mean = vec(Statistics.mean(data, dims=2))
        l = lines!(ax, data_mean, z_km, color = color, label=sim_label)

        local data_std = vec(Statistics.std(data, dims=2))

        local lower = Point2.(data_mean .- 2data_std, z_km)
        local upper = Point2.(data_mean .+ 2data_std, z_km)
        band!(ax, lower, upper, color = (color, 0.2))
    end

    # prettify
    ylims!(ax, 0, z_max / 1e3)
    if ind ∈ CartesianIndex.(1:rows, 3)
        axislegend(ax)
    end
end

fig[0, :] = Label(fig, case_name)
save("profiles_$case_name.png", fig)
