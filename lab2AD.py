import urllib.request 
import pandas as pd
import os
from datetime import datetime

def download_vhi_data(province_id, start_year=1981, end_year=2024):
    url = f"https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?country=UKR&provinceID={province_id}&year1={start_year}&year2={end_year}&type=Mean"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'vhi_id_{province_id}_{timestamp}.csv'
    if os.path.exists(filename):
        print(f"Файл для області {province_id} вже існує: {filename}. Пропускаємо завантаження.")
        return
    
    try:
        vhi_url = urllib.request.urlopen(url)
        with open(filename, 'wb') as out:
            out.write(vhi_url.read())
        print(f"VHI дата для області {province_id} завантажена і збережена як {filename}")
    except Exception as e:
        print(f"Помилка для області {province_id}: {e}")

for province_id in range(1, 26):  
    download_vhi_data(province_id)

def read_vhi_data(directory):
    data_frames = []
    for filename in os.listdir(directory):
        if filename.startswith('vhi_id_') and filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            try:
                province_id = int(filename.split('_')[2])
            except (IndexError, ValueError) as e:
                print(f"Невірний формат імені файлу: {filename}. Видаляємо цей файл.")
                os.remove(filepath) 
                continue
            df = pd.read_csv(filepath, index_col=False, header=1)
            df['provinceID'] = province_id  
            
            data_frames.append(df)    
    return pd.concat(data_frames, ignore_index=True)

data_directory = '.' 
vhi_data = read_vhi_data(data_directory)
vhi_data.columns = vhi_data.columns.str.strip().str.replace('<br>', '', regex=True)
vhi_data['year'] = vhi_data['year'].astype(str).str.extract(r'(\d+)')
vhi_data = vhi_data.dropna(subset=['year']).copy() 
vhi_data['year'] = vhi_data['year'].astype(int) 
print("Стовпці у фреймі:")
print(vhi_data.columns)

# Словник
province_mapping = {
    1: "Вінницька", 2: "Волинська", 3: "Дніпропетровська", 4: "Донецька",
    5: "Житомирська", 6: "Закарпатська", 7: "Запорізька", 8: "Івано-Франківська",
    9: "Київська", 10: "Кіровоградська", 11: "Луганська", 12: "Львівська",
    13: "Миколаївська", 14: "Одеська", 15: "Полтавська", 16: "Рівненська",
    17: "Сумська", 18: "Тернопільська", 19: "Харківська", 20: "Херсонська",
    21: "Хмельницька", 22: "Черкаська", 23: "Чернівецька", 24: "Чернігівська",
    25: "Республіка Крим"
}

def replace_province_indices(df, mapping):
    if 'provinceID' in df.columns:
        df['province'] = df['provinceID'].map(mapping) 
    else:
        print("Помилка: стовпець 'provinceID' не знайдено у фреймі даних.")
    return df

vhi_data = replace_province_indices(vhi_data, province_mapping)

def analyze_vhi_data(df, province, year):
    province_data = df[(df['province'] == province) & (df['year'] == year)]
    if not province_data.empty:
        print("-"*70)
        print(f"Область: {province}, Рік: {year}")
        print(f"Мін VHI: {province_data['VHI'].min()}, Макс VHI: {province_data['VHI'].max()}, Серднє: {province_data['VHI'].mean()}, Медіана VHI: {province_data['VHI'].median()}")
        print("-"*70)
    else:
        print(f"Нема інформації для {province}обл in {year}")

def user_input_for_analysis(df):
    print("Доступні області:")
    for idx, province in province_mapping.items():
        print(f"{idx}: {province}")
    
    province_id = int(input("Введіть номер області (1-25): "))
    year = int(input("Введіть рік: "))
    
    if province_id not in province_mapping:
        print("Невірний номер області.")
        return
    
    province = province_mapping[province_id]
    analyze_vhi_data(df, province, year)

user_input_for_analysis(vhi_data)

def display_vhi_for_range(df):
    print("Доступні області:")
    for idx, province in province_mapping.items():
        print(f"{idx}: {province}")
    
    provinces_input = input("Введіть номери областей через кому (наприклад, 1,3,5): ")
    provinces_list = [int(x.strip()) for x in provinces_input.split(',')]
    
    start_year = int(input("Введіть початковий рік: "))
    end_year = int(input("Введіть кінцевий рік: "))
    
    invalid_provinces = [p for p in provinces_list if p not in province_mapping]
    if invalid_provinces:
        print(f"Невірні області: {', '.join(map(str, invalid_provinces))}.")
        return
    
    filtered_data = df[(df['provinceID'].isin(provinces_list)) & 
                        (df['year'].between(start_year, end_year))]

    year_difference = end_year - start_year
    row_limit = 52 + (year_difference * 52)
    
    if not filtered_data.empty:
        print("-"*70)
        for province_id in provinces_list:
            province_name = province_mapping[province_id]
            print(f"Ряд VHI для області {province_name} з {start_year} по {end_year}:")
            province_data = filtered_data[filtered_data['provinceID'] == province_id]
            province_data_limited = province_data.head(row_limit)
            print(province_data_limited[['year', 'VHI']].to_string(index=False))
            print("-"*70)
    else:
        print(f"Немає даних для вказаних областей або років.")

display_vhi_for_range(vhi_data)
def find_extreme_droughts_user_input(df, mapping):
    print("\n Аналіз екстремальних посух в Україні ")
    print("=" * 70)
    try:
        percent_threshold = float(input("Введіть відсоток областей для визначення екстремальних посух (наприклад, 20): ").replace(',', '.'))
        if percent_threshold <= 0 or percent_threshold > 100:
            print("\n Відсоток має бути в межах від 1 до 100.\n")
            return
    except ValueError:
        print("\n Невірний формат числа. Спробуйте ще раз.\n")
        return

    total_regions = 25
    threshold_regions = max(1, int(total_regions * percent_threshold / 100)) 
    print(f"\n Шукаємо роки, коли більше {percent_threshold:.1f}% областей (тобто {threshold_regions}+) постраждали від посухи (VHI < 15)...")
    print("=" * 70)

    drought_years = []
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        drought_regions = year_data[year_data['VHI'] < 15]['provinceID'].unique()

        if len(drought_regions) >= threshold_regions:
            affected_regions = [mapping[r] for r in drought_regions if r in mapping]
            drought_years.append({
                'year': year,
                'affected_count': len(drought_regions),
                'regions': affected_regions
            })

    if drought_years:
        print(f"\nЗнайдено {len(drought_years)} рік(ів) з екстремальними посухами!\n")
        for record in drought_years:
            print(f"📅 Рік: {record['year']} | 🌍 Уражено областей: {record['affected_count']}")
            print("📍 Області: " + ', '.join(record['regions']))
            print("-" * 70)
    else:
        print("\n❗ Посухи, що уразили більше зазначеного відсотка областей, не знайдено.")

find_extreme_droughts_user_input(vhi_data, province_mapping)


