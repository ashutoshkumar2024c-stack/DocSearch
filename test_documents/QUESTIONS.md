# Test pack: what to upload and what to ask

Upload all 11 files at once (sidebar > Process), then open **How it was parsed** to see which tool the router picked for each, and ask the questions below.

## 1. Files and the tool you should see

| File | What it is | Expected route | Tool |
|---|---|---|---|
| `01_leave_policy_digital.pdf` | Digital PDF (headings, paragraph, table) | Text layer (native) | pdfplumber |
| `02_invoice_scanned.pdf` | Scanned PDF, slightly skewed, noisy, ruled table | OCR | Tesseract + table detection |
| `03_receipt_photo.jpg` | Phone photo of a receipt: rotated, dim, uneven light | OCR (poor-quality path) | Tesseract with enhancement |
| `04_meeting_minutes.docx` | Word document with a table | Parser | python-docx |
| `05_product_launch.pptx` | PowerPoint, 4 slides, one table | Parser | python-pptx |
| `06_sales_data.xlsx` | Excel, two sheets | Parser | openpyxl |
| `07_students.csv` | CSV table | Parser | csv |
| `08_library_faq.html` | Web page (headings, list) | Parser | beautifulsoup |
| `09_lab_safety.md` | Markdown notes | Parser | text |
| `10_enrollment_chart.png` | Bar-chart image | OCR sparse-text mode (vision model if installed) | Tesseract, or Qwen3-VL with Ollama |
| `11_mixed_report.pdf` | 4-page PDF: native, scan, chart page, scanned table | Per page: native / OCR / hybrid / OCR | pdfplumber + Tesseract |

## 2. Questions with known answers

Each answer was checked by running the pipeline. Answers come from the file named in the first column.

| File | Question | Correct answer contains |
|---|---|---|
| `01_leave_policy_digital.pdf` | How many days of paid annual leave do full-time employees get? | 24 |
| `01_leave_policy_digital.pdf` | How many days of maternity leave are given? | 182 |
| `01_leave_policy_digital.pdf` | What are the types of leave? | Sick, Paternity |
| `01_leave_policy_digital.pdf` | What does the policy say about sick leave? | Sick, 12 |
| `01_leave_policy_digital.pdf` | How far in advance must leave be requested? | 5 working days |
| `02_invoice_scanned.pdf` | What is the invoice number? | 0417 |
| `02_invoice_scanned.pdf` | Who is the invoice billed to? | Sunrise |
| `02_invoice_scanned.pdf` | What is the grand total of the invoice? | 18,998 |
| `02_invoice_scanned.pdf` | How many wireless mouse were ordered? | 8 |
| `03_receipt_photo.jpg` | What is the total paid? | 540 |
| `03_receipt_photo.jpg` | How much was the sandwich? | 240 |
| `04_meeting_minutes.docx` | When will the annual tech fest be held? | 12 December |
| `04_meeting_minutes.docx` | Who is responsible for booking the auditorium? | Neha |
| `04_meeting_minutes.docx` | What is the approved budget for the event? | 4,50,000 |
| `05_product_launch.pptx` | What is the goal for the number of active users? | 10,000 |
| `05_product_launch.pptx` | How much is the development budget? | 38 |
| `05_product_launch.pptx` | What are the risks of the project? | payment gateway |
| `06_sales_data.xlsx` | How many units were sold in the West region? | 1430 |
| `06_sales_data.xlsx` | What is the Q4 target for the North region? | 600000 |
| `07_students.csv` | What is Kavya Nair's CGPA? | 8.9 |
| `07_students.csv` | Which department is Imran Shaikh in? | Electronics |
| `08_library_faq.html` | How many books can students borrow? | 4 books |
| `08_library_faq.html` | What is the late fee for overdue books? | Rs 5 |
| `09_lab_safety.md` | What should I do if a chemical spills on my skin? | 15 minutes |
| `09_lab_safety.md` | What is the emergency contact number? | 4455 |
| `10_enrollment_chart.png` | How many students are enrolled in CSE? | 420 |
| `10_enrollment_chart.png` | What is the enrollment of the Civil department? | 180 |
| `10_enrollment_chart.png` | What is the enrollment of the Biotech department? | 350 |
| `11_mixed_report.pdf` | How much was recovered from Blue Ocean Ltd? | 1,75,500 |
| `11_mixed_report.pdf` | What was the haemoglobin result? | 10.2 |
| `11_mixed_report.pdf` | What was the renewal rate in the West region? | 91 |

## 3. Questions the app must refuse

Neither fact is in any file, so the answer should be *I couldn't find this in the uploaded documents.*

- What is the capital of France?
- Who is the principal of Greenfield Institute?

## 4. Questions that need the language model (Mistral or Ollama)

The offline quoted-passage mode cannot do these; a written answer can.

- Which region sold the most units in Q3? (West, 1430)
- Who has the highest CGPA in the student list? (Ananya Iyer, 9.1)
- What is the total cost of the Aurora launch? (12 + 38 + 20 = Rs 70 lakh)
- Compare the leave a new father and a new mother can take. (15 days vs 182 days)
- Summarise what the tech fest meeting decided.

## 5. Follow-up chain (tests chat memory)

1. What is the Q4 target for the North region?  (600000)
2. And for the South region?  (480000)

## Notes

- The chart image is read by OCR in sparse mode; values are paired with their labels by position. With the Qwen3-VL vision model installed the router uses it instead and also describes the chart.
- Regenerate the files any time with `python tests/make_testpack.py`.
