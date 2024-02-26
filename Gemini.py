from flask import Flask, request, jsonify
#from pdfminer.high_level import extract_text
from flask import Flask, request, jsonify
#from pdfminer.high_level import extract_text
import re
from flask import Flask, render_template
import csv
import defusedxml
#import json
from flask import Flask, send_file
import json
import re
import os
import time  # Added for timestamp
import uuid  # Added for unique identifier
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_extraction_chain_pydantic
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.llms import GooglePalm, OpenAI
from flask_cors import CORS
from langchain.llms import GooglePalm
from PIL import Image

from typing import Optional
from pydantic import BaseModel
from langchain.chains import create_extraction_chain_pydantic
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.llms import GooglePalm, OpenAI
#import pytesseract
import cv2
from langchain.llms import HuggingFaceHub
import pandas as pd
#from asposecells.api import Workbook, JsonLoadOptions
#pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Replace with the path to your

api_key = 'AIzaSyDifpKp5EwKvKT1ygGTPlG4J5UIGmp00vA' 

from doctr.io import DocumentFile
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(google_api_key=api_key,model="gemini-pro")

# llm = GooglePalm(google_api_key=api_key, temperature=0)  # GooglePalm model
## Invoice details 
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.llms import GooglePalm, OpenAI

## Invoice details 
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.llms import GooglePalm, OpenAI

## Invoice details 
invoice_info = ResponseSchema(
        name="Invoice number",
        description="In this include Invoice number, Invoice number can be in any form example : Inv no, Order no, etc",
    )
order_number_info = ResponseSchema(
    name="Order number",
    description="Details related to the order number, such as the unique identifier for an order. Example : order no, Order number",
)

Invoice_Date = ResponseSchema(
        name="Invoice Date",
        description="In this will show invoice date",
    )
Due_Date = ResponseSchema(
        name="Due Date",
        description="In this will show Due Date if you not found due  date so just set 'None',"
    )

order_date_info = ResponseSchema(
    name="Order date",
    description="Details related to the order date, such as the date when an order was placed.",
)

total_refund_amount = ResponseSchema(
    name="refund amount",
    description="Details related to the refund_amount",
)
"""Currency_symbol = ResponseSchema(
    name="Currency Symbol",
    # description="Idenntify currency in invoice and extract with the name and sign"
    description="Examples: Input: Image containing '$' or '$45.75' or '$ 323.00' Output: USD, Confidence: 0.80,Input: '€45.75' or '€ 45.75' Output: EUR, Confidence: 0.80,Input: '£1000' ot '£ 1000' Output: GBP, Confidence: 0.80,Input: '₹ 20.50' or '₹20.50'Output: INR, Confidence: 0.80",
)"""
company_informations = ResponseSchema(
    name="Company Information",
    description="""Extracts Company's information from the document.
    field_definitions={
        "Name": str,
        "Address": str,
        "Fax": str,
        "Phone": int,
        "Email": str,
        "GST":str,
        "VAT Number": str,
        "Bankgiro": int,
        "Plusgiro": int,
        "Site URL": str
    }
    """

)


shipping_information = ResponseSchema(
    name="Shipping Information",
    description = """Extracts shipping information from the invoice. Shipping information is included only when found in the text; otherwise, it is set to 'none'."

    example={
        "Recipient Name": "str",
        "Shipping Address": "str",
        "Shipping Method": "str",
        "Tracking Number": "str",
        "mobile number":"int"
    }
    """
)
company_information_schema = ResponseSchema(
    name="Company Information",
    description="Retrieve details related to Company's Information, which may include key details about a company or organization. Extract and provide values for the following fields: Name, Company Type, Registration Number, Industry, Contact Information, Website URL and other relevant details. Only provide values that are found within the Company Information section or any other source related to sellers Company.",
)

Billing_Information = ResponseSchema(
        name="Billing Information",
        description="""Retrieve details related to Billing Information in the invoice. Extract and provide values for the following fields: "Name," "Phone," "Payment method," "Address," "Apartment Type,Due Date  Only provide the values that are found within the Billing Information section or any other related source related to seller in the invoice.
        example : {
        "Name" ,
        "Phone",
        "Payment method",
        "Address", 
        "Apartment Type",
        "Due Date"},"""
    )
Product_informations = ResponseSchema(
        name="Products information",
        description="Please provide details related to the Products list in a dictionary format. Extract and provide values for the following fields: 'Name,' 'SKU,' ''Price,' 'delivery date' ,'discount',' 'delivery Time' ,'tracking url','Quantity',and 'Subtotal' for each item. The JSON data should be structured as a dictionary with a key 'Items' that contains a list of dictionaries, each representing an item with these specified fields.",
)
other_information_schema = ResponseSchema(
    name="Other Information",
    description="Extract and provide miscellaneous invoice details, including Sub-total, Delivery fee, Tax, Service fee, Tips, Order total, and GST. Include only the values found within the Other Information section or source provided."
)

output_parser = StructuredOutputParser.from_response_schemas(
    [invoice_info,order_number_info,Invoice_Date,Due_Date,order_date_info,total_refund_amount, shipping_information, company_information_schema, Billing_Information,Product_informations,other_information_schema]
)

response_format = output_parser.get_format_instructions()
print(response_format)

prompt = ChatPromptTemplate.from_template("Please provide a valid JSON format, ensuring that the JSON data does not contain the '$' sign. JSON should follow the standard syntax and structure to be processed correctly. {processed_text} \n {response_format}"if response_format else "none")


app = Flask(__name__)

from doctr.models import ocr_predictor

model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
from doctr.io import DocumentFile



class InvoiceException(Exception):
    pass

class InvalidJsonException(Exception):
    pass

class ExcelUpdateIssue(Exception):
    pass

def check_invoice(dictionary):
    try:
        invoice_number = dictionary.get("Invoice number")
        order_number = dictionary.get("Order number")
        if (invoice_number is None or invoice_number == "") and (order_number is None or order_number == ""):
            raise InvoiceException("Invoice Number or Order Number is not present.")
    except KeyError:
        raise InvoiceException(" Invoice Number or Order Number is not present.")

def validate_json(data):
    try:
        json.loads(data)
    except json.JSONDecodeError:
        raise InvalidJsonException(f" Not a valid Invoice")
    
def remove_none_values(data):
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if value != "None" and value != "":
                new_value = remove_none_values(value)
                if new_value is not None:
                    new_data[key] = new_value
        return new_data
    elif isinstance(data, list):
        new_list = [remove_none_values(item) for item in data]
        return [item for item in new_list if item is not None]
    else:
        return data

# function for formattiing json

def flatten_dict(data, parent_key='', sep='.'):
    items = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
 
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
 
        elif isinstance(value, list):
            for i, item in enumerate(value):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    items.update(flatten_dict(item, list_key, sep=sep))
                else:
                    items[list_key] = item if item not in ('', 'None') else None
 
        else:
            items[new_key] = value if value not in ('', 'None') else None
 
    return {k: v for k, v in items.items() if v is not None}


#  function to format json

def flatten_json(json_obj, parent_key='', sep='.'):
    flattened = {}
    for key, value in json_obj.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_json(value, new_key, sep=sep))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                flattened.update(flatten_json(item, f"{new_key}{sep}{i}", sep=sep))
        else:
            flattened[new_key] = value
    return flattened

# function to convert dict to list
def DictToList(input_list):
    output_list = []
    
    # Iterate through the input list
    for entry in input_list:
        new_entries = []
    
        # Iterate through key-value pairs in the entry
        for key, value in entry.items():
            # Check if the key starts with 'Products information.Items.'
            if key.startswith('Products information.Items.'):
                # Extract the section number and subkey
                section_number, subkey = key.split('.')[-2:]
                new_key = f'Products information.Items.{subkey}'
                
                # Create a new entry for each section
                if len(new_entries) <= int(section_number):
                    new_entries.append({new_key: value})
                else:
                    new_entries[int(section_number)][new_key] = value
            else:
                # Add other key-value pairs as is
                new_entries.append({key: value})
    
        # Append the modified entries to the output list
        output_list.extend(new_entries)
    return output_list


# function to add list to excel

def add_data_to_excel(file_path, data_list):
    try:
        # Read the existing Excel file into a DataFrame or create a new one if it doesn't exist
        try:
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            df = pd.DataFrame()

        # Initialize row_index before the loop
        row_index = 0 if not df.empty else len(df)

        # Iterate over the list of dictionaries and add data to the DataFrame
        a = len(df)
        for data_dict in data_list:
            for key, value in data_dict.items():
                # Check if the key (column name) exists in the DataFrame
                if key in df.columns:
                    # Find the first row without data in the column
                    row_index = max(a, df[key].index[df[key].isnull()].min()) if df[key].isnull().any() else len(df)
                    # Set the value in the corresponding row and column
                    df.at[row_index, key] = value
                else:
                    # If the key doesn't exist, check for similar columns
                    matching_columns = [col for col in df.columns if key in col]
                    if matching_columns:
                        # If similar columns exist, add the value to each matching column
                        for col in matching_columns:
                            df[col] = df[col].combine_first(pd.Series([None] * row_index + [value] + [None] * (len(df) - row_index - 1)))
                    else:
                        # If no similar column exists, add a new column
                        df[key] = pd.Series([None] * row_index + [value] + [None] * (len(df) - row_index - 1))

        # Save the updated DataFrame back to the Excel file
        df.to_excel(file_path, index=False)
        return file_path  # Return the updated file path

    except Exception as e:
        raise e  # Propagate the exception for handling elsewhere

def create_document(file_path):
    # Check the file type based on the file extension
    if file_path.lower().endswith('.pdf'):
        # Create a DocumentFile from a PDF
        document = DocumentFile.from_pdf(file_path)
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
        # Create a DocumentFile from an image
        document = DocumentFile.from_images(file_path)
    else:
        raise ValueError("Unsupported file type. Supported types: PDF, JPEG, PNG, GIF")
    
    return document

CORS(app)
UPLOAD_FOLDER = r'UPLOAD_FOLDER'
JSON_FOLDER = r'JSON_FOLDER'
CSV_FOLDER = r'CSV_FOLDER'
IMAGE_UPLOAD=r'IMAGE_UPLOAD'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_FOLDER'] = JSON_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/v1/img-text', methods=['POST'])
def extract_text_from_img():
    files = request.files.getlist('image')

    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        if files :
            for files in files:
               
                fname = files.filename
                name, extension = fname.rsplit(".", 1)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                unique_identifier = str(uuid.uuid4().hex[:6])
                new_filename = f"{name}_{timestamp}_{unique_identifier}.{extension}"
                # Define file paths using os.path.join
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

                files.save(file_path)
                
                document = create_document(file_path) 
                result = model(document)
                #image=result.show(document)
                
                text=result.render()
                # Perform basic processing to remove '\n'
                processed_text = " ".join(text.split("\n"))
                recipe = processed_text

                formated_prompt = prompt.format(**{"processed_text": recipe, "response_format": output_parser.get_format_instructions()})
                response_palm = llm.invoke(formated_prompt)    #Gemini
                # response_palm = llm(formated_prompt) # google palm
                response_palm = response_palm.content
            
                
                cleaned_json = re.sub(r'\s{4,}', '', response_palm)
                cleaned_json = re.sub(r'\$', '', response_palm)
                cleaned_json = re.sub(r',\s+', ', ', cleaned_json)
                cleaned_json = re.sub(r'\n', '', cleaned_json)
                cleaned_json = re.sub(r'\t', '', cleaned_json)
                cleaned_json = re.sub(r'<json>|<\/json>', '', cleaned_json)
                cleaned_json = re.sub(r'```json', '', cleaned_json)
                cleaned_json = re.sub(r'```', '', cleaned_json)
                cleaned_json = re.sub(r'\?', '', cleaned_json)
            
                data_dict = json.loads(cleaned_json)


                #input_list = result_list
                #print(input_list)
                #output_list = []
                
                   
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8181, debug=True)