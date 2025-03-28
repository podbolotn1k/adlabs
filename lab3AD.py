import os 
import pandas as pd
import glob
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="VHI Data Analysis", 
    page_icon=":chart_with_upwards_trend:", 
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f0f2f6;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.stTitle {
    color: #2c3e50;
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 20px;
}
.stHeader {
    color: #34495e;
    background-color: #ecf0f1;
    padding: 10px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #e0e6f0;
    border-radius: 10px;
    padding: 5px;
}
.stTabs [data-baseweb="tab"] {
    transition: all 0.3s ease;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background-color: #3498db;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = "csvfiles"
os.makedirs(DATA_DIR, exist_ok=True)

allreg = {
    1: "Вінницька", 2: "Волинська", 3: "Дніпропетровська", 4: "Донецька",
    5: "Житомирська", 6: "Закарпатська", 7: "Запорізька", 8: "Івано-Франківська",
    9: "Київська", 10: "Кіровоградська", 11: "Луганська", 12: "Львівська",
    13: "Миколаївська", 14: "Одеська", 15: "Полтавська", 16: "Рівненська",
    17: "Сумська", 18: "Тернопільська", 19: "Харківська", 20: "Херсонська",
    21: "Хмельницька", 22: "Черкаська", 23: "Чернівецька", 24: "Чернігівська",
    25: "Республіка Крим"
}

def read_vhi_data(directory):
    data_frames = []
    if not os.path.exists(directory):
        st.error(f"Директорія {directory} не існує!")
        return pd.DataFrame()
    
    for filename in os.listdir(directory):
        if filename.startswith('vhi_id_') and filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            try:
                province_id = int(filename.split('_')[2].split('.')[0])  
                df = pd.read_csv(filepath, index_col=False, header=1, delimiter=',')
                df['provinceID'] = province_id  
                data_frames.append(df)
            except (IndexError, ValueError) as e:
                print(f"Невірний формат імені файлу: {filename}. Помилка: {e}")
                continue
            except pd.errors.EmptyDataError:
                print(f"Файл {filename} порожній або має невірний формат.")
                continue
            except Exception as e:
                print(f"Помилка при обробці файлу {filename}: {e}")
                continue
    
    if not data_frames:
        print("Не знайдено жодного відповідного файлу для обробки.")
        return pd.DataFrame()
    
    try:
        df_combined = pd.concat(data_frames, ignore_index=True)
        df_combined.columns = [col.lower().strip().replace('<br>', '') for col in df_combined.columns]
        
        return df_combined
    except Exception as e:
        print(f"Помилка при об'єднанні даних: {e}")
        return pd.DataFrame()

def read_data_to_dataframe(directory):
    df = read_vhi_data(directory)
    
    if df.empty:
        return pd.DataFrame()
    
    if 'year' in df.columns:
        df['year'] = df['year'].astype(str).str.extract(r'(\d+)')
        df = df.dropna(subset=['year']).copy() 
        df['year'] = df['year'].astype(int)
    else:
        print("Попередження: стовпець 'year' не знайдено")
    
    if 'week' in df.columns:
        df['week'] = df['week'].astype(int)
    
    if 'provinceid' in df.columns:
        df = replace_province_indices(df, allreg)
    else:
        print("Попередження: стовпець 'provinceID' не знайдено")
    
    return df

def replace_province_indices(df, mapping):
    df['province'] = df['provinceid'].map(mapping)
    return df

def main():
    st.markdown('<h1 class="stTitle">🌍 Аналіз Вегетаційного Здоров\'я Регіонів</h1>', unsafe_allow_html=True)
    
    df = read_data_to_dataframe(DATA_DIR)  

    if df.empty:
        st.error("Дані відсутні. Будь ласка, перевірте:")
        st.error("- Чи папка 'data' містить файли у форматі 'vhi_id_XX.csv'?")
        st.error("- Чи мають файли правильний формат даних?")
        return
    
    if 'provinceid' not in df.columns:
        st.error("У даних відсутній стовпець 'provinceID'. Перевірте формат вхідних файлів.")
        return
    
    filter, graf = st.columns([1, 6])
    
    with filter:
        st.markdown('<h2 class="stHeader">🔍 Фільтри</h2>', unsafe_allow_html=True)

        analysis_columns = [col for col in df.columns if col in ['vhi', 'vci', 'tci']]
        
        analysis_type = st.selectbox(
            "Оберіть тип аналізу",
            analysis_columns,
            index=0,
            help="Виберіть показник для аналізу: VHI (Вегетаційний Індекс Здоров'я), VCI (Вегетаційний Кондиційний Індекс), або TCI (Температурний Кондиційний Індекс)"
        )
    
        region_options = {k: v for k, v in allreg.items() if k in df['provinceid'].unique()}
        selected_region = st.selectbox(
            "Оберіть регіон",
            options=list(region_options.keys()),
            format_func=lambda x: region_options[x],
            help="Виберіть область України для деталізованого аналізу"
        )
        
        min_week, max_week = st.slider(
            "Оберіть діапазон тижнів",
            min_value=int(df['week'].min()),
            max_value=int(df['week'].max()),
            value=(1, 52),
            help="Виберіть тижневий діапазон для аналізу"
        )
        
        min_year, max_year = st.slider(
            "Оберіть діапазон років",
            min_value=int(df['year'].min()),
            max_value=int(df['year'].max()),
            value=(int(df['year'].min()), int(df['year'].max())),
            help="Виберіть часовий діапазон для аналізу"
        )

    filtered_df = df[
        (df['provinceid'] == selected_region) & 
        (df['week'] >= min_week) & (df['week'] <= max_week) & 
        (df['year'] >= min_year) & (df['year'] <= max_year)
    ].copy()
    
    with graf:
        tab1, tab2, tab3 = st.tabs(["📊 Таблиця даних", "📈 Часовий ряд", "🌐 Порівняння регіонів"])
        
        with tab1:
            st.subheader("Таблиця даних")

            sort_column = st.selectbox("Сортувати за:", filtered_df.columns, index=0)
            sort_order = st.radio("Порядок сортування:", ["Зростання", "Спадання"], horizontal=True)
            filtered_df = filtered_df.sort_values(by=sort_column, ascending=(sort_order == "Зростання"))
            st.dataframe(filtered_df, use_container_width=True)

        
        with tab2:
            if not filtered_df.empty:
                st.subheader(f"{analysis_type.upper()} по роках ({region_options[selected_region]})")
                
                plot_df = filtered_df.copy()
                plot_df['date'] = pd.to_datetime(plot_df['year'].astype(str) + '-W' + plot_df['week'].astype(str) + '-1', format='%Y-W%W-%w')
                
                plt.style.use('default') 
                fig, ax = plt.subplots(figsize=(12, 6))
                plt.plot(plot_df['date'], plot_df[analysis_type], label=analysis_type.upper(), color='#3498db', linewidth=2)
                plt.title(f"{analysis_type.upper()} по роках", fontsize=15, fontweight='bold')
                plt.xlabel("Дата", fontsize=12)
                plt.ylabel(analysis_type.upper(), fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                st.pyplot(fig)
            else:
                st.warning("Немає даних для відображення графіку")
        
        with tab3:
            st.subheader(f"{analysis_type.upper()} у різних регіонах")
            
            if 'province' in df.columns:
                compare_stats = df.groupby(['provinceid', 'province'])[analysis_type].mean().reset_index()
                
                plt.style.use('default')  
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = plt.bar(compare_stats['province'], compare_stats[analysis_type], color='#2ecc71')
                plt.title(f"Середнє {analysis_type.upper()} по регіонах", fontsize=15, fontweight='bold')
                plt.xlabel("Регіон", fontsize=12)
                plt.ylabel(f"Середнє {analysis_type.upper()}", fontsize=12)
                plt.xticks(rotation=90)
                
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}', 
                             ha='center', va='bottom', fontsize=9)
                
                st.pyplot(fig)
            else:
                st.warning("Відсутні дані про регіони для порівняння")

if __name__ == "__main__":
    main()