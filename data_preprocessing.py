import pandas as pd
import argparse
import os
from tqdm import tqdm


def clean_csv(input_file, min_text_length=20, min_likes=500, min_year=2020, max_data=1000, save_cleaned_csv=False):
    df = pd.read_csv(input_file)
    
    cleaned_df = pd.DataFrame(columns=df.columns)
    data = 0
    
    with tqdm(total=max_data) as pbar:
        for index, row in df.iterrows():
            review_text = row['review_text']
            review_likes = row['review_likes']
            review_timestamp = row['review_timestamp']
            review_year = int(review_timestamp.split('-')[0])
            
            if review_year < min_year:
                continue
            
            if len(str(review_text)) >= min_text_length and review_likes > min_likes:
                cleaned_df = pd.concat([cleaned_df, pd.DataFrame([row], columns=df.columns)])
                data += 1
                pbar.update(1)
            
            if data >= max_data:
                break
        print('total data:', data)
    if save_cleaned_csv:
        cleaned_df.to_csv('cleaned_data.csv', index=False, sep='\t')
    return cleaned_df

def convert_to_txt(input_file, output_folder):
    def rating_to_text(rating):
        ratings = {
            1: 'very bad',
            2: 'bad',
            3: 'average',
            4: 'good',
            5: 'very good'
        }
        return ratings.get(rating, 'unknown rating')

    if isinstance(input_file, pd.DataFrame):
        df = input_file
        print('input_file is a pandas DataFrame')
    else:
        df = pd.read_csv(input_file, sep='\t')
    print('Converting to text files...')
    for index, row in tqdm(df.iterrows()):
        rating_text = rating_to_text(row['review_rating'])
        data = f"rating: {rating_text}, review: {row['review_text']}"
        
        with open(os.path.join(output_folder, f'review_{index}.txt'), 'w') as txt_file:
            txt_file.write(data)
    print("Conversion to text files completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean CSV file')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('--output_folder', type=str, default='dataset_review',help='Path to the output txt_file')
    parser.add_argument('--min_text_length', type=int, default=20, help='Minimum length of review text')
    parser.add_argument('--min_likes', type=int, default=500, help='Minimum number of review likes')
    parser.add_argument('--min_year', type=int, default=2020, help='Minimum review year')
    parser.add_argument('--max_data', type=int, default=1000, help='Maximum number of rows in the output CSV file')

    args = parser.parse_args()

    cleaned_df = clean_csv(args.input_file, args.min_text_length, args.min_likes, args.min_year, args.max_data)
    os.makedirs(args.output_folder, exist_ok=True)
    convert_to_txt(cleaned_df, args.output_folder)