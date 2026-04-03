with open(r"C:\Users\Mykola\Downloads\FOP.xml", 'r', encoding='utf-8') as f:
    for _ in range(15):  # Виведемо перші 15 рядків
        print(f.readline().strip())



'https://data.gov.ua/dataset/a1799820-195b-4982-8141-6e84f58103e7'

import pandas as pd
import xml.etree.ElementTree as ET
import os

def xml_to_csv_chunked(xml_path, csv_path, record_tag='SUBJECT', chunk_size=50000):
    """
    Парсить великий XML файл та зберігає його частинами у CSV.
    """
    print(f"Починаємо обробку файлу: {xml_path}")
    
    context = ET.iterparse(xml_path, events=('end',))
    
    batch = []
    chunk_counter = 1
    total_records = 0
    write_header = not os.path.exists(csv_path) 
    
    for event, elem in context:
        # Шукаємо завершення тегу <SUBJECT>
        if elem.tag == record_tag:
            row_data = {}
            
            # Збираємо всі "листочки" (NAME, STAN, RECORD тощо)
            for child in elem:
                # Якщо тег порожній (наприклад <EXCHANGE_DATA/>), запишемо None
                row_data[child.tag] = child.text.strip() if child.text else None
                
            batch.append(row_data)
            total_records += 1
            
            # Звільняємо пам'ять! (відрізаємо оброблену гілку)
            elem.clear()
            
            # Якщо назбирали достатньо даних — скидаємо на диск
            if len(batch) >= chunk_size:
                df = pd.DataFrame(batch)
                df.to_csv(csv_path, mode='a', index=False, header=write_header, encoding='utf-8')
                
                print(f"Збережено чанк {chunk_counter}: +{chunk_size} записів (Загалом: {total_records})")
                
                batch = [] 
                write_header = False 
                chunk_counter += 1
                
    # Зберігаємо "хвостик" даних, який залишився після останнього повного батчу
    if batch:
        df = pd.DataFrame(batch)
        df.to_csv(csv_path, mode='a', index=False, header=write_header, encoding='utf-8')
        print(f"Збережено фінальний чанк {chunk_counter}: +{len(batch)} записів (Загалом: {total_records})")
        
    print(f"\nОбробку завершено! Успішно перетворено {total_records} записів.")

# === Запуск ===
input_xml = r"C:\Users\Mykola\Downloads\UO.xml"
output_csv = "UO_parsed.csv"

# Викликаємо функцію з правильним тегом SUBJECT
xml_to_csv_chunked(input_xml, output_csv, record_tag='SUBJECT', chunk_size=50000)