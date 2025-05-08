import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from deep_translator import GoogleTranslator 
import sys


PDF_FILE_PATH = "python-for-everybody.pdf" 
PERSIST_DIRECTORY = "db_vector_store" 
OUTPUT_FILENAME = "translated_summary_fa.txt"
QUERY = "What are the basics of Python variables?" 
SOURCE_LANGUAGE = 'en' 
TARGET_LANGUAGE = 'fa' 
SIMILARITY_K = 3


try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass 


if not os.path.exists(PDF_FILE_PATH):
    print(f"خطا: فایل PDF در مسیر '{PDF_FILE_PATH}' یافت نشد.")
    exit()

print(f"۱. در حال بارگذاری سند: {PDF_FILE_PATH}...")

loader = PyPDFLoader(PDF_FILE_PATH)
pages = loader.load() 
print(f"   تعداد صفحات بارگذاری شده: {len(pages)}")
if not pages:
    print("خطا: هیچ صفحه‌ای از PDF بارگذاری نشد. فایل ممکن است خالی یا خراب باشد.")
    exit()

print("۲. در حال تقسیم متن به قطعات کوچکتر...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

texts = text_splitter.split_documents(pages)
print(f"   تعداد کل قطعات متن: {len(texts)}")
if not texts:
    print("خطا: هیچ متنی برای پردازش پس از تقسیم‌بندی وجود ندارد.")
    exit()

print("۳. در حال تولید Embedding و ذخیره/بارگذاری از پایگاه داده Chroma...")

embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")


db = None
if os.path.exists(PERSIST_DIRECTORY) and os.path.isdir(PERSIST_DIRECTORY) and len(os.listdir(PERSIST_DIRECTORY)) > 0:
    print(f"   پایگاه داده از قبل در '{PERSIST_DIRECTORY}' موجود است. در حال بارگذاری...")
    try:
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        print("   بارگذاری پایگاه داده تکمیل شد.")
       
        try:
            db.similarity_search("test", k=1)
            print("   تست بارگذاری پایگاه داده موفق بود.")
        except Exception as load_test_e:
            print(f"   هشدار: پایگاه داده بارگذاری شد اما در تست اولیه خطا وجود دارد: {load_test_e}")
            print("   ممکن است نیاز به ساخت مجدد پایگاه داده باشد اگر جستجوها ناموفق بود.")

    except Exception as e:
        print(f"خطا در بارگذاری پایگاه داده از '{PERSIST_DIRECTORY}': {e}")
        print("   تلاش برای ساخت پایگاه داده جدید...")
        db = None 


if db is None:
    print(f"   در حال ساخت پایگاه داده جدید و ذخیره در '{PERSIST_DIRECTORY}'...")
    try:
        db = Chroma.from_documents(documents=texts,
                                   embedding=embeddings,
                                   persist_directory=PERSIST_DIRECTORY)
        print("   ساخت و ذخیره پایگاه داده جدید تکمیل شد.")
    except Exception as create_e:
        print(f"خطا در ساخت پایگاه داده Chroma: {create_e}")
        print("لطفا خطا را بررسی کنید. ممکن است مشکل از داده‌ها یا دسترسی به پوشه باشد.")
        exit()

if db is None:
     print("خطا: ایجاد یا بارگذاری پایگاه داده ناموفق بود. برنامه متوقف می‌شود.")
     exit()

print(f"\n۴. در حال جستجوی مشابهت برای کوئری: '{QUERY}'...")

try:
    similar_docs = db.similarity_search(QUERY, k=SIMILARITY_K)
    print(f"   تعداد {len(similar_docs)} نتیجه مشابه یافت شد.")
    if not similar_docs:
        print("   هشدار: هیچ نتیجه مشابهی برای کوئری شما یافت نشد.")
except Exception as search_e:
    print(f"خطا در هنگام جستجوی مشابهت: {search_e}")
    similar_docs = []


translated_texts = []
if similar_docs: 
    print(f"\n۵. در حال ترجمه نتایج یافت شده از '{SOURCE_LANGUAGE}' به '{TARGET_LANGUAGE}' با deep-translator...")
    try:
        
        translator = GoogleTranslator(source=SOURCE_LANGUAGE, target=TARGET_LANGUAGE)
    except Exception as translator_init_e:
         print(f"خطا در ساخت مترجم deep-translator: {translator_init_e}")
         translator = None 

    if translator:
        for i, doc in enumerate(similar_docs):
            original_text = doc.page_content
            print(f"   در حال ترجمه قطعه {i+1}/{len(similar_docs)}...")
            try:
                
                translated = translator.translate(original_text)
                if translated: 
                    translated_texts.append(translated)
                else:
                    print(f"     هشدار: ترجمه برای قطعه {i+1} نتیجه‌ای نداشت.")
                    translated_texts.append(f"[ترجمه ناموفق یا خالی]:\n{original_text}\n")
            except Exception as e:
                print(f"خطا در ترجمه قطعه {i+1} با deep-translator: {e}")
                translated_texts.append(f"[خطا در ترجمه - {e}]:\n{original_text}\n")
    else:
        print("   مترجم بارگذاری نشد، از مرحله ترجمه صرف‌نظر می‌شود.")
        
        for doc in similar_docs:
             translated_texts.append(f"[ترجمه انجام نشد - متن اصلی]:\n{doc.page_content}\n")



print("\n۶. در حال آماده‌سازی متن نهایی برای ذخیره...")
separator = "\n\n" + "="*30 + "\n\n" 
full_output_text = separator.join(translated_texts)


output_content = f"نتایج جستجو و (در صورت امکان) ترجمه برای کوئری: '{QUERY}'\n"
output_content += f"زبان مبدا سند: {SOURCE_LANGUAGE}, زبان مقصد ترجمه: {TARGET_LANGUAGE}\n"
output_content += "="*50 + "\n\n"
if not translated_texts:
     output_content += "هیچ نتیجه‌ای برای نمایش یا ترجمه یافت نشد.\n"
else:
    output_content += full_output_text


print(f"\n۷. در حال ذخیره خروجی در فایل: '{OUTPUT_FILENAME}'...")
try:
   
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        f.write(output_content)
    print(f"   خروجی با موفقیت در فایل '{OUTPUT_FILENAME}' ذخیره شد.")
    print("\nمی‌توانید فایل را باز کرده و نتایج را مشاهده کنید.")
except IOError as e:
    print(f"خطا در هنگام نوشتن در فایل '{OUTPUT_FILENAME}': {e}")
except Exception as e:
    print(f"یک خطای غیرمنتظره در ذخیره فایل رخ داد: {e}")