# Helper funcs

function get_time(ds, var)
    if haskey(ds, var)
        return ds[var][:]
    elseif haskey(ds.group["timeseries"], var)
        return ds.group["timeseries"][var][:]
    end
    error("No key for $var found in the nc file.")
end

"""
    name_aliases()

Returns a `Dict` containing:
 - a key (`String`), which we consider to be our core variable name
 - values (`Tuple` of `String`s), which we consider to be aliases of our core variable name
"""
function name_aliases()
    dict = Dict(
        "zc" => ("z_half",),
        "zf" => ("z",),
        "α0_c" => ("alpha0_half",),
        "α0_f" => ("alpha0",),
        "p0_c" => ("p0_half",),
        "p0_f" => ("p0",),
        "ρ0_c" => ("rho0_half",),
        "ρ0_f" => ("rho0",),
        "updraft_area" => ("updraft_fraction",),
        "updraft_thetal" => ("updraft_thetali",),
        "thetal_mean" => ("thetali_mean", "theta_mean"),
        "total_flux_h" => ("resolved_z_flux_thetali", "resolved_z_flux_theta"),
        "total_flux_qt" => ("resolved_z_flux_qt", "qt_flux_z"),
        "u_mean" => ("u_translational_mean",),
        "v_mean" => ("v_translational_mean",),
        "tke_mean" => ("tke_nd_mean",),
        "total_flux_s" => ("s_flux_z",),
    )
    return dict
end

"""
    get_nc_data(ds::NCDatasets.Dataset, var::String)

Returns the data for variable `var`, trying first its aliases
defined in `name_aliases`, in the `ds::NCDatasets.Dataset`.
"""
function get_nc_data(ds, var::String)
    dict = name_aliases()
    key_options = haskey(dict, var) ? (dict[var]..., var) : (var,)

    for key in key_options
        if haskey(ds, key)
            return ds[key]
        else
            for group_option in ["profiles", "reference", "timeseries"]
                haskey(ds.group, group_option) || continue
                if haskey(ds.group[group_option], key)
                    return ds.group[group_option][key]
                end
            end
        end
    end
    @warn "Variable $var not in dataset. Returning `nothing`."
    return nothing
end



"""
    is_face_variable(filename::String, var_name::String)
    is_face_variable(ds::NC.Dataset, var_name::String)

A `Bool` indicating whether the given variable is defined on faces, or not.
"""
function is_face_variable(filename::String, var_name::String)
    NCDataset(filename) do ds
        is_face_variable(ds, var_name)
    end
end

function is_face_variable(ds::NC.Dataset, var_name::String)

    # PyCLES cell face variables
    pycles_face_vars = ["w_mean", "w_mean2", "w_mean3"]
    for group_option in ["profiles", "reference", "timeseries"]
        haskey(ds.group, group_option) || continue
        if haskey(ds.group[group_option], var_name)
            var_dims = dimnames(ds.group[group_option][var_name])
            if ("zc" in var_dims) | ("z_half" in var_dims)
                return false
            elseif ("zf" in var_dims) | (var_name in pycles_face_vars)
                return true
            elseif ("z" in var_dims) # "Inconsistent" PyCLES variables
                return false
            else
                error("Variable $var_name does not contain a vertical coordinate.")
            end
        end
    end
end

"""
    get_les_names(y_names::Vector{String}, filename::String)
    get_les_names(m::ReferenceModel, filename::String)

Returns the aliases of the variables actually present in the nc
file (`filename`) corresponding to SCM variables `y_names`.
"""
function get_les_names(y_names::Vector{String}, filename::String)::Vector{String}
    dict = name_aliases()
    y_alias_groups = [haskey(dict, var) ? (dict[var]..., var) : (var,) for var in y_names]
    return [find_alias(aliases, filename) for aliases in y_alias_groups]
end

function get_les_names(y_names::Vector{String}, ds::NC.Dataset)::Vector{String}
    dict = name_aliases()
    y_alias_groups = [haskey(dict, var) ? (dict[var]..., var) : (var,) for var in y_names]
    return [find_alias(aliases, ds) for aliases in y_alias_groups]
end

"""
    find_alias(aliases::Tuple{Vararg{String}}, filename::String)
    find_alias(aliases::Tuple{Vararg{String}}, ds::NC.Dataset)

Finds the alias present in an NCDataset from a list of possible aliases.
"""
function find_alias(aliases::Tuple{Vararg{String}}, filename::String)
    NC.NCDataset(filename) do ds
        find_alias(aliases, ds)
    end
end

function find_alias(aliases::Tuple{Vararg{String}}, ds::NC.Dataset)
    for alias in aliases
        if haskey(ds, alias)
            return alias
        else
            for group_option in ["profiles", "reference", "timeseries"]
                haskey(ds.group, group_option) || continue
                if haskey(ds.group[group_option], alias)
                    return alias
                end
            end
        end
    end
    error("None of the aliases $aliases found in the dataset.")
end
