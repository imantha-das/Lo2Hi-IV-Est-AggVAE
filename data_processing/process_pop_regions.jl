# ------------------------------ Process Regions ----------------------------- #
# We only have population at High resolution regions (state level)
# To get population at census level, we will aggregate states within census regions

using CSV
using DataFrames
using XLSX: readtable
using StatsBase: countmap

# Load population data
state_data = readtable("data/raw/SCPRC-EST2024-18+POP.xlsx", "SCPRC-EST2024-18+POP", first_row = 3) |> DataFrame
# We dont need the third column which describes the percentage of under 18 year olds
select!(state_data, names(state_data)[1:2])
rename!(state_data, [:state, :tot_popn])
# We have a missing row, just after the column names that need to be dropped
dropmissing!(state_data)
# This first 4 rows are not needed as they contain population of a more coarser set of census regions
drop_census_divisions = ["United States","Northeast","Midwest","South","West"]
filter!(row -> !(row.state in drop_census_divisions),state_data)
# There are is "." in front of every region that needs to be removed
transform!(state_data, :state => ByRow(x -> begin
    s = strip(string(x))
    startswith(s,".") ? s[2:end] : s
end) => :state)
# There are several states that are NOT part of the US mainlnd : Drop them
not_us_mainland = ["Alaska", "Hawaii", "American Samoa", "Puerto Rico",
                    "United States Virgin Islands", "Guam", "Commonwealth of the Northern Mariana Islands"]
filter!(row -> !(row.state in not_us_mainland), state_data)

# Load state and census region dataframe
state_census_divisions = CSV.read("data/raw/us_census_bureau_regions_and_divisions.csv", DataFrame)
select!(state_census_divisions, ["State", "Division"])
rename!(state_census_divisions, [:state, :division])
filter!(row -> !(row.state in not_us_mainland), state_census_divisions)              

# Join the two dataframe, so that we have popultion for both census dics and states
state_census_popn = innerjoin(state_data, state_census_divisions, on = :state)
# Groupby census div and aggregate to get total popn at a census divison
census_popn = combine(groupby(state_census_popn, :division), :tot_popn => sum => :tot_popn)

CSV.write("data/processed/high/state_popn.csv", state_data)
CSV.write("data/processed/low/census_popn.csv", census_popn)

