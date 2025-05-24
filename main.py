import pandas as pd
from preprocess import clean_text
from ranker import rank_resumes

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=['Resume'], inplace=True)
    df['cleaned_resume'] = df['Resume'].apply(clean_text)
    return df

def main():
    csv_path = 'resume_dataset/UpdatedResumeDataSet.csv'
    print('[INFO] Loading and preprocessing data...')
    df = load_data(csv_path)

    job_description = input('\nEnter job description:\n')
    cleaned_jd = clean_text(job_description)

    print('\n[INFO] Ranking resumes based on similarity...\n')
    resume_texts = df['cleaned_resume'].tolist()
    ranked_indices, scores = rank_resumes(resume_texts, cleaned_jd)

    top_k = 5
    print(f'Top {top_k} resumes for the job description:\n')
    for i in range(top_k):
        idx = ranked_indices[i]
        print(f'{i+1}. Score: {scores[i]:.4f}')
        print(df.iloc[idx]["Resume"][:400], '\n---\n')

if __name__ == '__main__':
    main()