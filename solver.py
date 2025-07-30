from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, lpSum, LpInteger, PULP_CBC_CMD

# Sets
plants = ['Waterloo', 'Kingston']
dcs = ['Saskatoon', 'Edmonton', 'Calgary', 'Hamilton', 'Moncton', 'Montreal']
stores = ['Toronto', 'Montreal', 'Winnipeg', 'Vancouver', 'Edmonton', 'North York',
'Regina', 'St. John’s', 'Halifax', 'Ottawa']


# Parameters
demand = {'Toronto':2265, 'Montreal':1471, 'Winnipeg':300.2, 'Vancouver':991.1,
           'Edmonton':520.3, 'North York':217.9,'Regina':89.90, 'St. John’s':43.40, 
           'Halifax': 169.0, 'Ottawa': 529.2}

fixed_cost = {'Saskatoon':0,'Edmonton':0, 'Calgary':0, 'Hamilton':0, 
              'Moncton':0,'Montreal':0}

rail_cost = {
    ('Waterloo', 'Saskatoon'): 0,
    ('Waterloo', 'Edmonton'): 0,
    ('Waterloo', 'Calgary'): 0,
    ('Waterloo', 'Hamilton'): 0,
    ('Waterloo', 'Moncton'): 0,
    ('Waterloo', 'Montreal'): 0,

    ('Kingston', 'Saskatoon'): 0,
    ('Kingston', 'Edmonton'): 0,
    ('Kingston', 'Calgary'): 0,
    ('Kingston', 'Hamilton'): 0,
    ('Kingston', 'Moncton'): 0,
    ('Kingston', 'Montreal'): 0
}

truck_cost = {
    ('Saskatoon', 'Toronto'): 0,
    ('Saskatoon', 'Montreal'): 0,
    ('Saskatoon', 'Winnipeg'): 0,
    ('Saskatoon', 'Vancouver'): 0,
    ('Saskatoon', 'Edmonton'): 0,
    ('Saskatoon', 'North York'): 0,
    ('Saskatoon', 'Regina'): 0,
    ('Saskatoon', 'St. John’s'): 0,
    ('Saskatoon', 'Halifax'): 0,
    ('Saskatoon', 'Ottawa'): 0,

    ('Edmonton', 'Toronto'): 0,
    ('Edmonton', 'Montreal'): 0,
    ('Edmonton', 'Winnipeg'): 0,
    ('Edmonton', 'Vancouver'): 0,
    ('Edmonton', 'Edmonton'): 0,
    ('Edmonton', 'North York'): 0,
    ('Edmonton', 'Regina'): 0,
    ('Edmonton', 'St. John’s'): 0,
    ('Edmonton', 'Halifax'): 0,
    ('Edmonton', 'Ottawa'): 0,

    ('Calgary', 'Toronto'): 0,
    ('Calgary', 'Montreal'): 0,
    ('Calgary', 'Winnipeg'): 0,
    ('Calgary', 'Vancouver'): 0,
    ('Calgary', 'Edmonton'): 0,
    ('Calgary', 'North York'): 0,
    ('Calgary', 'Regina'): 0,
    ('Calgary', 'St. John’s'): 0,
    ('Calgary', 'Halifax'): 0,
    ('Calgary', 'Ottawa'): 0,

    ('Hamilton', 'Toronto'): 0,
    ('Hamilton', 'Montreal'): 0,
    ('Hamilton', 'Winnipeg'): 0,
    ('Hamilton', 'Vancouver'): 0,
    ('Hamilton', 'Edmonton'): 0,
    ('Hamilton', 'North York'): 0,
    ('Hamilton', 'Regina'): 0,
    ('Hamilton', 'St. John’s'): 0,
    ('Hamilton', 'Halifax'): 0,
    ('Hamilton', 'Ottawa'): 0,

    ('Moncton', 'Toronto'): 0,
    ('Moncton', 'Montreal'): 0,
    ('Moncton', 'Winnipeg'): 0,
    ('Moncton', 'Vancouver'): 0,
    ('Moncton', 'Edmonton'): 0,
    ('Moncton', 'North York'): 0,
    ('Moncton', 'Regina'): 0,
    ('Moncton', 'St. John’s'): 0,
    ('Moncton', 'Halifax'): 0,
    ('Moncton', 'Ottawa'): 0,

    ('Montreal', 'Toronto'): 0,
    ('Montreal', 'Montreal'): 0,
    ('Montreal', 'Winnipeg'): 0,
    ('Montreal', 'Vancouver'): 0,
    ('Montreal', 'Edmonton'): 0,
    ('Montreal', 'North York'): 0,
    ('Montreal', 'Regina'): 0,
    ('Montreal', 'St. John’s'): 0,
    ('Montreal', 'Halifax'): 0,
    ('Montreal', 'Ottawa'): 0
}

# Model
model = pulp.LpProblem("Facility_Location_Distribution", pulp.LpMinimize)

# Decision Variables
y = pulp.LpVariable.dicts("OpenDC", dcs, cat='Binary')
x = pulp.LpVariable.dicts("Assign", [(i, j) for i in dcs for j in stores], cat='Binary')
z = pulp.LpVariable.dicts("Ship", [(p, i) for p in plants for i in dcs], lowBound=0, cat='Integer')

# Objective
model += (
    pulp.lpSum(rail_cost[p, i] * z[p, i] for p in plants for i in dcs) +
    pulp.lpSum(truck_cost[i, j] * demand[j] * x[i, j] for i in dcs for j in stores) +
    pulp.lpSum(fixed_cost[i] * y[i] for i in dcs)
)

# Constraints
# 1. Open only 2 DCs
model += pulp.lpSum(y[i] for i in dcs) == 2

# 2. Each store assigned to one DC
for j in stores:
    model += pulp.lpSum(x[i, j] for i in dcs) == 1

# 3. Store can only be assigned if DC is open
for i in dcs:
    for j in stores:
        model += x[i, j] <= y[i]

# 4. Flow into each DC must meet demand of stores assigned to it
for i in dcs:
    model += pulp.lpSum(z[p, i] for p in plants) == pulp.lpSum(demand[j] * x[i, j] for j in stores)

# Solve
model.solve()
print("Status:", pulp.LpStatus[model.status])
print("Total Cost = $", pulp.value(model.objective))


# Results
print("\nOpened Distribution Centers:")
for i in dcs:
    if y[i].varValue == 1:
        print(f"  - {i}")

print("\nStore Assignments:")
for (i, j) in x:
    if x[i, j].varValue == 1:
        print(f"  - {j} assigned to {i}")

print("\nPlant to DC Shipments:")
for (p, i) in z:
    if z[p, i].varValue > 0:
        print(f"  - {p} → {i}: {z[p, i].varValue} units")