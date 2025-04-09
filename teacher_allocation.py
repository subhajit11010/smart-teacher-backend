import pandas as pd
import math
import numpy as np
import copy
import joblib
scaler = joblib.load('./scaler.pkl')
kmeans = joblib.load('./kmeans_model.pkl')
data = {}
df = ""
df_test_scaled = ""
teacher_threshold_per_subject = 2
total_students = []
total_teachers = []
actual_PTR = []
standard_PTR = []
current_teachers = []
required_teachers = []
additional_teachers_needed = []
school_subject_teacher_req = {}
all_subjects = []
suggestions = {}

column_order = [
    'total_teacher', 'male', 'female', 'regular', 'contract', 'part_time',
    'total_teacher_trained_computer', 'class_taught_pr', 'class_taught_upr',
    'class_taught_pr_upr', 'class_taught_sec_only', 'class_taught_hsec_only',
    'class_taught_upr_sec', 'class_taught_sec_hsec',
    'teacher_involve_non_training_assignment', 'standard_PTR',
    'sch_enroll_primary', 'sch_enroll_upper_primary', 'sch_enroll_secondary',
    'sch_enroll_higher_secondary', 'total_students', 'actual_PTR',
    'school_category_Primary', 'school_category_Primary & Upper Primary',
    'school_category_Primary, Upper Primary & Secondary',
    'school_category_Primary, Upper Primary, Secondary & Higher Secondary',
    'school_category_Upper Primary', 'school_category_Upper Primary & Secondary',
    'school_category_Upper Primary, Secondary & Higher Secondary',
    'school_type_Boys', 'school_type_Co-educational', 'school_type_Girls'
]

def categorize_teacher_availability(row):
    if row['actual_PTR'] > 30:
        return 'Scarcity'
    elif row['actual_PTR'] < 15:
        return 'Surplus'
    else:
        return 'Normal'

def create_df(schools):
    schools_dup = copy.deepcopy(schools)
    for school in schools_dup:
        school.pop('age', None)
        school.pop('gender', None)
        school.pop('location', None)
        school.pop('qualifications', None)
        school.pop('teachers', None)
        school.pop('willinness_to_reallocate', None)
        school.pop('subjects_taught', None)
        school.pop('sch_name', None)
        # resolving the 'Co-Ed and 'Co-educational issue'
        if school['school_type'] == 'Co-Ed':
            school['school_type'] = 'Co-educational'
        school["class_taught_sec_only"] = int(school["class_taught_sec"])
        school.pop('class_taught_sec', None)
        school["class_taught_hsec_only"] = int(school["class_taught_hsec"])
        school.pop('class_taught_hsec', None)

    df = pd.DataFrame(schools_dup)
    df.set_index("unique_id", inplace=True)
    categorical_cols = ["school_type", "school_category"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    school_categories = [
    "Primary",
    "Primary & Upper Primary",
    "Primary, Upper Primary & Secondary",
    "Primary, Upper Primary, Secondary & Higher Secondary",
    "Upper Primary",
    "Upper Primary & Secondary",
    "Upper Primary, Secondary & Higher Secondary"
    ]
    school_types = [
        "Boys",
        "Girls",
        "Co-educational"
    ]

    for category in school_categories:
        col_name = f"school_category_{category}"
        if col_name not in df.columns:
            df[col_name] = 0

    for type in school_types:
        col_name = f"school_type_{type}"
        if col_name not in df.columns:
            df[col_name] = 0

    bool_columns = ['school_category_Primary', 'school_category_Primary & Upper Primary', 'school_category_Primary, Upper Primary & Secondary', 'school_category_Primary, Upper Primary, Secondary & Higher Secondary', 'school_category_Upper Primary', 'school_category_Upper Primary & Secondary', 'school_category_Upper Primary, Secondary & Higher Secondary', 'school_type_Boys', 'school_type_Co-educational', 'school_type_Girls']

    df[bool_columns] = df[bool_columns].astype(int)

    numeric_cols = ['male', 'regular', 'contract', 'part_time', 'class_taught_pr', 'class_taught_upr',
    'class_taught_pr_upr', 'class_taught_sec_only',
    'class_taught_hsec_only', 'class_taught_upr_sec',
    'class_taught_sec_hsec', 'teacher_involve_non_training_assignment',
    'standard_PTR', 'sch_enroll_primary', 'sch_enroll_higher_secondary', 'total_students',
    'actual_PTR',
    'sch_enroll_secondary',
    'sch_enroll_upper_primary',
    'total_teacher',
    'total_teacher_trained_computer',
    'female'
    ]
    df_test_scaled = df.copy()
    df_test_scaled[numeric_cols] = scaler.transform(df[numeric_cols])

    df = df.reindex(columns=column_order)
    df_test_scaled = df_test_scaled.reindex(columns=column_order)
    df['cluster'] = kmeans.predict(df_test_scaled)
    df['teacher_availability'] = df.apply(categorize_teacher_availability, axis=1)

    total_students = df["total_students"].tolist()
    total_teachers = df["total_teacher"].tolist()
    actual_PTR = df["actual_PTR"].tolist()
    standard_PTR = df["standard_PTR"].tolist()

    current_teachers = [x / y if y != 0 else 0 for x, y in zip(total_students, actual_PTR)]
    required_teachers = [x / y if y != 0 else 0 for x, y in zip(total_students, standard_PTR)]

    additional_teachers_needed = [math.ceil(required - current) for current, required in zip(current_teachers, required_teachers)]
    print(df.columns)

    
    # df.to_csv("df.csv", index= True)
    # df_test_scaled.to_csv("df_test_scaled.csv", index = True)

def count_positive_and_negative_elem(sub, schools_with_sub):
    teacher_required_in_sub=[]
    for sch in schools_with_sub:
        teacher_required_in_sub.append(school_subject_teacher_req[sch][sub])

    positive_count = 0
    negative_count = 0
    for element in teacher_required_in_sub:
        if element > 0:
            positive_count += 1
        elif element < 0:
            negative_count += 1
    return positive_count, negative_count



def analyze_and_reallocate(sub, schools_with_sub):
    suggestions = []
    allocated_teachers = set()

    for sch1 in schools_with_sub:
        for sch2 in schools_with_sub:
            if sch1 != sch2:
                if (school_subject_teacher_req[sch1][sub] > 0 and school_subject_teacher_req[sch2][sub] < 0):
                    teacher_indices_in_data = [i for i, x in enumerate(data[sch2]["subjects_taught"]) if x == sub]
                    for index in teacher_indices_in_data:
                        if data[sch2]["willinness_to_reallocate"][index] == "Yes":
                            if(data[sch2]["gender"][index].lower() in data[sch1]["school_type"].lower()):
                                teacher_to_be_allocated = data[sch2]["teachers"][index]
                                teacher_qualification = data[sch2]["qualifications"][index]
                                if teacher_to_be_allocated not in allocated_teachers:
                                    allocated_teachers.add(teacher_to_be_allocated)
                                    suggestions.append(
                                            f"Teacher {teacher_to_be_allocated} will be allocated to {data[sch1]["sch_name"]} from {data[sch2]["sch_name"]} "
                                            f"having a {teacher_qualification} degree in {sub}."
                                    )
                                    school_subject_teacher_req[sch1][sub] -= 1
                                    school_subject_teacher_req[sch2][sub] += 1
                                    positive_count, negative_count = count_positive_and_negative_elem(sub, schools_with_sub)
                                    if(positive_count == 0 or negative_count == 0):
                                        return suggestions

                elif (school_subject_teacher_req[sch1][sub] < 0 and school_subject_teacher_req[sch2][sub] > 0):
                    teacher_indices_in_data = [i for i, x in enumerate(data[sch1]["subjects_taught"]) if x == sub]
                    for index in teacher_indices_in_data:
                        if data[sch1]["willinness_to_reallocate"][index] == "Yes":
                            if(data[sch1]["gender"][index].lower() in data[sch2]["school_type"].lower()):
                                teacher_to_be_allocated = data[sch1]["teachers"][index]
                                teacher_qualification = data[sch1]["qualifications"][index]
                                if teacher_to_be_allocated not in allocated_teachers:
                                    allocated_teachers.add(teacher_to_be_allocated)
                                    suggestions.append(
                                                f"Teacher {teacher_to_be_allocated} will be allocated to {data[sch2]["sch_name"]} from {data[sch1]["sch_name"]} "
                                                f"having a {teacher_qualification} degree in {sub}."
                                    )
                                    school_subject_teacher_req[sch1][sub] += 1
                                    school_subject_teacher_req[sch2][sub] -= 1
                                    positive_count, negative_count = count_positive_and_negative_elem(sub, schools_with_sub)
                                    if(positive_count == 0 or negative_count == 0):
                                        return suggestions

def send_suggestions():
    for sub in all_subjects:
        schools_with_sub = []
        for sch in data:
            school = data[sch]
            if sub in school["subjects_taught"]:  # ✅ Check for ALL subjects dynamically
                schools_with_sub.append(sch)

        positive_count, negative_count = count_positive_and_negative_elem(sub, schools_with_sub)

        if positive_count == 0 or negative_count == 0:
            continue  # ✅ Skip if all schools either need or have excess teachers
        else:
            suggestions[sub] = analyze_and_reallocate(sub, schools_with_sub)


def process_input(schools):
    create_df(schools)
    for school in schools:
        school_type = ""
        if school["school_type"] == "Boys":
            school_type = "male"
        elif school["school_type"] == "Girls":
            school_type = "female"
        else:
            school_type = "male/female"

        data[school["unique_id"]] = {
            "sch_name": school["sch_name"],
            "teachers": school["teachers"],
            "subjects_taught": school["subjects_taught"],
            "qualifications": school["qualifications"],
            "willinness_to_reallocate": school["willinness_to_reallocate"],
            "gender": school["gender"],
            "age": school["age"],
            "school_type": school_type
        }
    
    for sch in data:
        school = data[sch]
        school_subject_teacher_req[sch] = {}
        subjects = school['subjects_taught']
        for subject in subjects:
            if subject not in school_subject_teacher_req[sch]:
            # school_subject_teacher_dist[sch][subject] = 1
                school_subject_teacher_req[sch][subject] = teacher_threshold_per_subject - 1 # these logic will be changed later because of the  dynamic nature of "teacher_threshold_per_subject"
            else:
            # school_subject_teacher_dist[sch][subject] += 1
                school_subject_teacher_req[sch][subject] -= 1
    
    for sch in data:        
        school = data[sch]
        subjects = school['subjects_taught']
        for subject in subjects:
            if subject not in all_subjects:
                all_subjects.append(subject)

    
    send_suggestions()
    return suggestions





