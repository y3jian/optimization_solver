import pulp
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, lpSum, LpInteger, PULP_CBC_CMD

# Sets
plants = ['Waterloo', 'Kingston']
dcs = ['Saskatoon', 'Edmonton', 'Thunder Bay', 'Hamilton', 'Moncton', 'Montreal']
stores = ['Toronto', 'Montreal', 'Winnipeg', 'Vancouver', 'Edmonton', 'North York',
'Regina', 'St. John’s', 'Halifax', 'Ottawa']


# Parameters
demand = {'Toronto':2265, 'Montreal':1471, 'Winnipeg':300.2, 'Vancouver':1391.1,
           'Edmonton':520.3, 'North York':217.9,'Regina':89.90, 'St. John’s':43.40, 
           'Halifax': 569.0, 'Ottawa': 529.2}

fixed_cost = {'Saskatoon':22440,'Edmonton':22970, 'Thunder Bay': 17580, 'Hamilton':18620, 
              'Moncton':17260,'Montreal':18180}

rail_cost = {
    ('Waterloo', 'Saskatoon'): 11333.7,
    ('Waterloo', 'Edmonton'): 13660.9,
    ('Waterloo', 'Thunder Bay'): 5245,
    ('Waterloo', 'Hamilton'): 731.21,
    ('Waterloo', 'Moncton'): 6728.5,
    ('Waterloo', 'Montreal'): 2805.3,

    ('Kingston', 'Saskatoon'): 12395.35,
    ('Kingston', 'Edmonton'): 14715.52,
    ('Kingston', 'Thunder Bay'):6383.99,
    ('Kingston', 'Hamilton'): 1406.17,
    ('Kingston', 'Moncton'): 5139.54,
    ('Kingston', 'Montreal'): 1216.33
}

truck_cost = {
    ('Saskatoon', 'Toronto'): 8.86,
    ('Saskatoon', 'Montreal'): 9.77,
    ('Saskatoon', 'Winnipeg'): 2.5,
    ('Saskatoon', 'Vancouver'): 5.05,
    ('Saskatoon', 'Edmonton'): 1.66,
    ('Saskatoon', 'North York'): 9.42,
    ('Saskatoon', 'Regina'): 0.84,
    ('Saskatoon', 'St. John’s'): 16.5,
    ('Saskatoon', 'Halifax'): 13.7,
    ('Saskatoon', 'Ottawa'): 9.14,

    ('Edmonton', 'Toronto'): 10.5,
    ('Edmonton', 'Montreal'): 11.41,
    ('Edmonton', 'Winnipeg'): 4.15,
    ('Edmonton', 'Vancouver'): 3.69,
    ('Edmonton', 'Edmonton'): 0,
    ('Edmonton', 'North York'): 10.5,
    ('Edmonton', 'Regina'): 2.49,
    ('Edmonton', 'St. John’s'): 18.2,
    ('Edmonton', 'Halifax'): 15.3,
    ('Edmonton', 'Ottawa'): 10.8,

    ('Thunder Bay', 'Toronto'): 4.42,
    ('Thunder Bay', 'Montreal'): 5.07,
    ('Thunder Bay', 'Winnipeg'): 2.24,
    ('Thunder Bay', 'Vancouver'): 9.6,
    ('Thunder Bay', 'Edmonton'): 6.42,
    ('Thunder Bay', 'North York'): 4.4,
    ('Thunder Bay', 'Regina'): 4.11,
    ('Thunder Bay', 'St. John’s'): 11.83,
    ('Thunder Bay', 'Halifax'): 9.01,
    ('Thunder Bay', 'Ottawa'): 4.66,

    ('Hamilton', 'Toronto'): 0.22,
    ('Hamilton', 'Montreal'): 2.01,
    ('Hamilton', 'Winnipeg'): 6.27,
    ('Hamilton', 'Vancouver'): 13.2,
    ('Hamilton', 'Edmonton'): 10.3,
    ('Hamilton', 'North York'): 0.31,
    ('Hamilton', 'Regina'): 7.82,
    ('Hamilton', 'St. John’s'): 8.84,
    ('Hamilton', 'Halifax'): 6.01,
    ('Hamilton', 'Ottawa'): 1.51,

    ('Moncton', 'Toronto'): 5.32,
    ('Moncton', 'Montreal'): 3.16,
    ('Moncton', 'Winnipeg'): 10.3,
    ('Moncton', 'Vancouver'): 17.6,
    ('Moncton', 'Edmonton'): 14.5,
    ('Moncton', 'North York'): 4.86,
    ('Moncton', 'Regina'): 12.2,
    ('Moncton', 'St. John’s'): 3.67,
    ('Moncton', 'Halifax'): 0.84,
    ('Moncton', 'Ottawa'): 3.75,

    ('Montreal', 'Toronto'): 1.73,
    ('Montreal', 'Montreal'): 0,
    ('Montreal', 'Winnipeg'): 7.23,
    ('Montreal', 'Vancouver'): 14.5,
    ('Montreal', 'Edmonton'): 11.4,
    ('Montreal', 'North York'): 1.7,
    ('Montreal', 'Regina'): 9.09,
    ('Montreal', 'St. John’s'): 6.8,
    ('Montreal', 'Halifax'): 3.97,
    ('Montreal', 'Ottawa'): 0.63
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