import bayespy as bp
import numpy as np
import csv
from colorama import init, Fore, Style

init(autoreset=True)

# Enumeration mappings
ageEnum = {'SuperSeniorCitizen': 0, 'SeniorCitizen': 1, 'MiddleAged': 2, 'Youth': 3, 'Teen': 4}
genderEnum = {'Male': 0, 'Female': 1}
familyHistoryEnum = {'Yes': 0, 'No': 1}
dietEnum = {'High': 0, 'Medium': 1, 'Low': 2}
lifeStyleEnum = {'Athlete': 0, 'Active': 1, 'Moderate': 2, 'Sedetary': 3}
cholesterolEnum = {'High': 0, 'BorderLine': 1, 'Normal': 2}
heartDiseaseEnum = {'Yes': 0, 'No': 1}

# Read the dataset
data = []
try:
    with open('heart-disease-dataset.csv') as csvfile:
        lines = csv.reader(csvfile, delimiter='\t')  # Use tab delimiter
        next(lines)  # Skip header
        for x in lines:
            if len(x) != 7:
                print(f"Skipping malformed row: {x}")
                continue
            try:
                data.append([
                    ageEnum[x[0].strip()],
                    genderEnum[x[1].strip()],
                    familyHistoryEnum[x[2].strip()],
                    dietEnum[x[3].strip()],
                    lifeStyleEnum[x[4].strip()],
                    cholesterolEnum[x[5].strip()],
                    heartDiseaseEnum[x[6].strip()]
                ])
            except KeyError as e:
                print(f"Skipping row with invalid value: {x} -- {e}")
except FileNotFoundError:
    print(Fore.RED + "Error: File 'heart_disease_data.csv' not found.")
    exit()

data = np.array(data)
N = len(data)

if N == 0:
    print(Fore.RED + "Dataset is empty or invalid. Please check the input file.")
    exit()

# BayesPy model
p_age = bp.nodes.Dirichlet(1.0 * np.ones(5))
age = bp.nodes.Categorical(p_age, plates=(N,))
age.observe(data[:, 0])

p_gender = bp.nodes.Dirichlet(1.0 * np.ones(2))
gender = bp.nodes.Categorical(p_gender, plates=(N,))
gender.observe(data[:, 1])

p_familyhistory = bp.nodes.Dirichlet(1.0 * np.ones(2))
familyhistory = bp.nodes.Categorical(p_familyhistory, plates=(N,))
familyhistory.observe(data[:, 2])

p_diet = bp.nodes.Dirichlet(1.0 * np.ones(3))
diet = bp.nodes.Categorical(p_diet, plates=(N,))
diet.observe(data[:, 3])

p_lifestyle = bp.nodes.Dirichlet(1.0 * np.ones(4))
lifestyle = bp.nodes.Categorical(p_lifestyle, plates=(N,))
lifestyle.observe(data[:, 4])

p_cholesterol = bp.nodes.Dirichlet(1.0 * np.ones(3))
cholesterol = bp.nodes.Categorical(p_cholesterol, plates=(N,))
cholesterol.observe(data[:, 5])

p_heartdisease = bp.nodes.Dirichlet(np.ones(2), plates=(5, 2, 2, 3, 4, 3))
heartdisease = bp.nodes.MultiMixture(
    [age, gender, familyhistory, diet, lifestyle, cholesterol],
    bp.nodes.Categorical,
    p_heartdisease
)
heartdisease.observe(data[:, 6])
p_heartdisease.update()

# Prediction loop
while True:
    print(Fore.CYAN + "\n--- Heart Disease Risk Prediction ---")
    try:
        input_age = int(input('Enter Age category ' + str(ageEnum) + ": "))
        input_gender = int(input('Enter Gender ' + str(genderEnum) + ": "))
        input_family = int(input('Enter FamilyHistory ' + str(familyHistoryEnum) + ": "))
        input_diet = int(input('Enter Diet ' + str(dietEnum) + ": "))
        input_lifestyle = int(input('Enter LifeStyle ' + str(lifeStyleEnum) + ": "))
        input_chol = int(input('Enter Cholesterol ' + str(cholesterolEnum) + ": "))

        prediction_node = bp.nodes.MultiMixture(
            [input_age, input_gender, input_family, input_diet, input_lifestyle, input_chol],
            bp.nodes.Categorical,
            p_heartdisease
        )
        prob = prediction_node.get_moments()[0][heartDiseaseEnum['Yes']]
        print(Fore.RED + "Probability of Heart Disease: {:.2f}%".format(prob * 100))

        cont = input("Enter 0 to Continue, 1 to Exit: ")
        if cont.strip() == "1":
            break
    except Exception as e:
        print(Fore.YELLOW + "Invalid input or error occurred: {}".format(e))
