# ---------------------------------------------------------------------------- #
#         Ensure all counties in geodataframe is the same as population        #
# Some of the counties in population dataframe are named differently to the shape
# file dataframe. These need to be ammended 
# The Connecticut region counties have been renamed since 2022. Overlapping 
# Population was redistributed based on constant weighting. This isnt accurate
# but simplifys the population redistribution. 
# ---------------------------------------------------------------------------- #

begin
    import Pkg 
    Pkg.activate("jl_env/lo2hi")
    import XLSX
    import CSV
    using DataFrames 
    using GeoDataFrames
    using StringDistances
    using StatsBase
end 

df_pop = CSV.read("data/raw/co-est2024-alldata.csv", DataFrame)
rename!(df_pop, "STATE" => :statefp, "COUNTY" => :countyfp, 
"STNAME" => :state, "CTYNAME" => :county, 
"POPESTIMATE2020" => :pop2020, "POPESTIMATE2021" => :pop2021,
"POPESTIMATE2022" => :pop2022, "POPESTIMATE2023" => :pop2023,
"POPESTIMATE2024" => :pop2024
)
select!(df_pop, :statefp, :countyfp, :state, :county, :pop2020,
:pop2021, :pop2022, :pop2023, :pop2024
)

gdf_county = GeoDataFrames.read("data/processed/gis/county_grid_v2/county_grid_v2.shp")
gdf_states = GeoDataFrames.read("data/processed/gis/high/us_state_divisions/us_state_divisions.shp")

st2fp_df = unique(df_pop[:, [:statefp, :state]])
#st2fp = Dict(length(string(k)) < 2 ? "0$k" : string(k) => v for (k,v) in zip(st2fp_df.statefp, st2fp_df.state))

# use dict to add state column to "gdf_county"

st2fp = Dict{String,String}()
for x in eachrow(st2fp_df)
    state_num = x.statefp 
    state_name = String(x.state) 
    state_num_str = lpad(string(state_num), 2, '0') 
    st2fp[state_num_str] = state_name
end

transform!(gdf_county, :statefp =>  ByRow(x -> st2fp[x]) => :state)
# remove rows where state and county are the same as thats the total of all counties 
filter!(row -> row.state != row.county, df_pop)
transform!(df_pop, :county => ByRow(x -> join(split(x)[1:end-1], " ")) => :county2)
select!(df_pop, Not(:county))
rename!(df_pop, :county2 => :county)
rename!(gdf_county, :name => :county)

uniq_county_states = unique(gdf_county.state) #49
uniq_county = unique(gdf_county.county) #1800
uniq_pop_states = unique(df_pop.state) #50
uniq_pop_county = unique(df_pop.county) #1837
uniq_states = unique(gdf_states.area)

symdiff(uniq_states, uniq_county_states)
non_overlap_states = symdiff(uniq_states, uniq_pop_states)
# So there are some states in pop dataframe that we must remove 
filter!(x -> x.state !="Alaska" , df_pop)
filter!(x -> x.state !="Hawaii", df_pop)

symdiff(uniq_states, unique(df_pop.state)) 
# We still have to add District of Columbia which isnt found in the df_pop
@doc"""
    There are some missing regions between population and county shape 
    dataframes 
"""->
function add_missing_regions!(df_pop)
    # Adds the district of columbia to the population dataframe 
    if "District of Columbia" ∉ df_pop.state
        new_row = (
        statefp = 11,   # DC's FIPS state code
        countyfp = 1,   # DC has county FIPS 001
        state = "District of Columbia",
        pop2020 = 689545,
        pop2021 = 670050,
        pop2022 = 671803,
        pop2023 = 678972,
        pop2024 = 684362,
        county = "District of Columbia"
    )
    push!(df_pop, new_row)
    end 
end

# We need to add District of Columbia 
add_missing_regions!(df_pop)

function find_missing_states_counties(county_df::DataFrame,pop_df::DataFrame,states::Vector{String})
    miss_counties = Dict()
  
    for state in states 
        tmp_county = filter(:state => ==(state), county_df)
        tmp_pop = filter(:state => ==(state), pop_df)
        non_overlap = symdiff(tmp_county.county, tmp_pop.county)
        @show non_overlap
        if length(non_overlap) != 0
            push!(miss_counties,state => ([tmp_county.county],[tmp_pop.county]))
        end
    end
    return miss_counties
end

uniq_gdf_states = gdf_states.area
miss_counties = find_missing_states_counties(gdf_county, df_pop, uniq_gdf_states)
# We need to sort these regions 
# New Mexico, "Nevada, "Massachusetts", "Connecticut", "District of Columbia
# ----------------------- Manually rename county names ----------------------- #
# Always aim to keep gdf_county unchanged while changing df_pop 

"Doña Ana" in gdf_county.county
"Do\xf1a Ana" in df_pop.county
# Dona Ana
replace!(gdf_county[!,:county], "Doña Ana" => "Dona Ana")
replace!(df_pop[!, :county], "Do\xf1a Ana" => "Dona Ana")

# Carson : We need to change df_pop county "Carson" to "Carson City" but texas also has County called "Carson" which must remain as it is 
"Carson City" in gdf_county.county
"Carson" in df_pop.county
df_pop[(df_pop.state .== "Nevada") .&& (df_pop.county .== "Carson"), :county] .= "Carson City"


# Region names for Connecticut was changed from 2022 onwards, we will keep the oldnames 
# as its better to change rows in population rather than shape file
connecticut_row1 = (statefp = 9, countyfp = 003, state = "Connecticut",   pop2020 = 964088, pop2021 = 971938, pop2022 = 975591, pop2023 = 983326, pop2024 =991508, county = "Hartford")
connecticut_row2 = (statefp = 9, countyfp = 001, state = "Connecticut",   pop2020 = 324397 + Int(round(0.5 * 618762)), pop2021 = 326799 + Int(round(0.5 * 623927)), pop2022 = 328131 + Int(round(0.5 * 625454)), pop2023 = 331300 + Int(round(0.5 * 630201)), pop2024 =335666 + Int(round(0.5 * 637013)), county = "Fairfield")
connecticut_row3 = (statefp = 9, countyfp = 005, state = "Connecticut",   pop2020 = 112199 + Int(round(0.5 * 618762)), pop2021 = 112952 + Int(round(0.5 * 623927)), pop2022 = 113289 + Int(round(0.5 * 625454)), pop2023 = 113538 + Int(round(0.5 * 630201)), pop2024 =114101 + Int(round(0.5 * 637013)), county = "Litchfield")
connecticut_row4 = (statefp = 9, countyfp = 009, state = "Connecticut",   pop2020 = 449055 + 564526, pop2021 = 452095 + 568961, pop2022 = 453868 + 569866, pop2023 = 457609 + 572919, pop2024 =462220 + 576718, county = "New Haven")
connecticut_row5 = (statefp = 9, countyfp = 015, state = "Connecticut",   pop2020 = Int(round(95231 * 0.5)), pop2021 = Int(round(95624 * 0.5)), pop2022 = Int(round(96169 * 0.5)), pop2023 = Int(round(96834 * 0.5)), pop2024 =Int(round(97701 * 0.5)), county = "Windham")
connecticut_row6 = (statefp = 9, countyfp = 013, state = "Connecticut",   pop2020 = Int(round(95231 * 0.5)), pop2021 = Int(round(95624 * 0.5)), pop2022 = Int(round(96169 * 0.5)), pop2023 = Int(round(96834 * 0.5)), pop2024 =Int(round(97701 * 0.5)), county = "Tolland")
connecticut_row7 = (statefp = 9, countyfp = 007, state = "Connecticut",   pop2020 = 173281, pop2021 = 175456, pop2022 = 176159, pop2023 = 176674, pop2024 = 177540, county = "Middlesex")
connecticut_row8 = (statefp = 9, countyfp = 011, state = "Connecticut",   pop2020 = 278379, pop2021 = 278855, pop2022 = 279398, pop2023 = 280622, pop2024 = 282602, county = "New London")
filter!(:state => !=("Connecticut"), df_pop)
for row in [connecticut_row1, connecticut_row2, connecticut_row3, connecticut_row4, connecticut_row5, connecticut_row6, connecticut_row7, connecticut_row8]
    push!(df_pop, row)
end

# Code to filter
filter(:county => ==("Carson City"), gdf_county)
filter(:county => ==("Carson City"), df_pop)
filter(:state => ==("Connecticut"),gdf_county)
filter(:state => ==("Connecticut"), df_pop)

miss_counties = find_missing_states_counties(gdf_county, df_pop, uniq_gdf_states)
@assert isempty(miss_counties)

df = innerjoin(gdf_county, df_pop, on = [:county, :state], makeunique = true)
save_root = "data/processed/gis/county_grid_pop_v2"
if !isdir(save_root)
    mkpath(save_root)
end 

GeoDataFrames.write(joinpath(save_root, "county_grid_pop_v2.shp"), df)

for col in eachcol(df)
    @show ismissing.(col) |> any
end

