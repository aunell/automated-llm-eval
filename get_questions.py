# Demo questions to make sure the pipeline works
# questions = [
#     "In individuals aged 18 to 85 years with hypertrophic cardiomyopathy who are treated with beta blockers only, calcium channels blockers only, or who switch from one to the other, is there a difference in time to developing atrial fibrillation, ventricular arrhythmia, heart failure or receiving a heart transplant?", 
#     "In patients at least 18 years old, and prescribed ibuprofen, is there any difference in peak blood glucose after treatment compared to patients prescribed acetaminophen?", 
#     "Among those with myotonic dystrophy, we are interested in whether treatments of mexiletine (typically used to ameliorate their muscle symptoms) would increase the risk of arrhythmia (both atrial and ventricular).", 
#     "Do outcomes differ for patients with syndromic vs non-syndromic causes of thoracic aortic aneurysms?", 
#     "Do patients who have an elevated incidental B12 lab go on to develop malignancy?"
# ]
def get_questions():
    run_number = 3
    file_path= "./questions.txt"
    with open(file_path, 'r') as file:
            # Read the file line by line and store each line as a string in a list
            questions = [line.strip() for line in file.readlines()]
    return questions