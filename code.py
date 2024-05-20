######################################################################
###  SET UP ###
######################################################################

import numpy as np
import cv2

import pandas as pd
import json

from pydantic import BaseModel, Field
from typing import List, Union
from datetime import date, time,datetime
from enum import Enum

from paddleocr import PaddleOCR

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

import uuid
import os

import plotly.express as px
import plotly.graph_objects as go

import base64

from PIL import Image
from io import BytesIO


######################################################################
###  EXMAPLES ###
######################################################################

example_cat_1= {
    "store": "HiperDino",
    "address": "9238-SD Bernardo de la torre",
    "city": "Tafira Baja",
    "phone": "928493638",
    "receipt_no": "2024/923813-00060866",
    "date": "15/04/2024",
    "time": "16:01",
    "items": [
        {"name": "FRESA TARINA 500 GR", "unit": 1, "price": 1.59, "amount": 1.59, "category": "fruits"},
        {"name": "HIPERDINO ACEITUNA R/ANCHOA LATA 350", "unit": 1, "price": 0.95, "amount": 0.95, "category": "canned_goods"},
        {"name": "DESPERADOS CERVEZA TOQUE TEQUILA BOT", "unit": 1, "price": 1.05, "amount": 1.05, "category": "beverages"},
        {"name": "HIPERDINO CENTRO JAMON SERRANO BODEG", "unit": 0.310, "price": 13.62, "amount": 4.22, "category": "protein_foods"},
        {"name": "MONTESANO JAMON COCIDO SELECCION KG", "unit": 0.308, "price": 8.74, "amount": 2.15, "category": "protein_foods"}
    ],
    "total": 9.96,
    "number_items": 5,
    "payment_method": "tarjeta"
}

example_cat_2 = {
    "store": "SPAR TAFIRA",
    "address": "C/. Bruno Naranjo DIAZ 9A-9B",
    "city": "Tafira Baja",
    "phone": "928 351 616",
    "receipt_no": "014\\002-18965",
    "date": "06/04/2024",
    "time": "15:23",
    "items": [
        {"name": "CLIPPER MANZ.1.5L.", "unit": 1, "price": 1.49, "amount": 1.49, "category": "beverages"},
        {"name": "PLATANO PRIMERA GR", "unit": 1.40, "price": 1.99, "amount": 2.79, "category": "fruits"},
        {"name": "MANZANA PINK LADY GR", "unit": 1, "price": 2.99, "amount": 2.99, "category": "fruits"},
        {"name": "SALSA.BARI.PES.GEN.1", "unit": 1, "price": 3.10, "amount": 3.10, "category": "condiments"},
        {"name": "GOFIO B.LUGAR MIL.FU", "unit": 1, "price": 1.85, "amount": 1.85, "category": "grains"},
        {"name": "ZUM.DISF.D.SIMON PIN", "unit": 1, "price": 1.75, "amount": 1.75, "category": "beverages"},
        {"name": "LECHE.GRNJ.FLR.UHT.", "unit": 1, "price": 1.15, "amount": 1.15, "category": "dairy"}
    ],
    "total": 15.12,
    "number_items": 7,
    "payment_method": "tarjeta"
}

example_cat_4 = {
    "store": "MERCADONA",
    "address": "AVDA. PINTOR FELO MONZON (C.C. 7 PALMAS) S/N",
    "city": "35019 LAS PALMAS DE GRAN CANARIA",
    "phone": "928411755",
    "receipt_no": "2185-013-6970Z2",
    "date": "03/04/2024",
    "time": "21:22",
    "items": [
        { "name": "DETERG HIPO COLONIA", "unit": 1, "price": 3.30, "amount": 3.30, "category": "household"},
        { "name": "SOLOMILLO POLLO CONG", "unit": 3, "price": 4.50, "amount": 13.50, "category": "protein_foods"},
        { "name": "JAMONCITO BARBACOA", "unit": 1, "price": 2.32, "amount": 2.32, "category": "protein_foods"},
        { "name": "JAMONCITO BARBACOA", "unit": 1, "price": 2.76, "amount": 2.76, "category": "protein_foods"},
        { "name": "NUEZ NATURAL", "unit": 1, "price": 2.00, "amount": 2.00, "category": "nuts_and_seeds"},
        { "name": "QUESU COTIAGE", "unit": 2, "price": 1.25, "amount": 2.50, "category": "dairy"},
        { "name": "POLLO ENTERO LIMPIO", "unit": 1, "price": 6.52, "amount": 6.52, "category": "protein_foods"},
        { "name": "PAPEL VEGETAL 30H", "unit": 1, "price": 1.70, "amount": 1.70, "category": "household"},
        { "name": "BEBIDA AVELLANAS", "unit": 1, "price": 1.30, "amount": 1.30, "category": "beverages"},
        { "name": "INFUSION DORMIR", "unit": 1, "price": 1.05, "amount": 1.05, "category": "beverages"},
        { "name": "LECHE DE COCO", "unit": 1, "price": 1.40, "amount": 1.40, "category": "beverages"},
        { "name": "QUESO UNTAR LIGHT", "unit": 1, "price": 1.35, "amount": 1.35, "category": "dairy"},
        { "name": "RULITO CABRA", "unit": 1, "price": 2.45, "amount": 2.45, "category": "dairy"},
        { "name": "GRIEGO LIGERO", "unit": 1, "price": 1.65, "amount": 1.65, "category": "dairy"},
        { "name": "BOLSA PLASTICO", "unit": 1, "price": 0.15, "amount": 0.15, "category": "household"}
        ],
    "total": 43.95,
    "number_items": 15,
    "payment_method": "tarjeta"
    }


receipt_texts_1 = [
    'HiperDino',
    'Las mcjores precios de Canarias',
    'DINOSOL SUPERMERCADOS. S.L',
    'C.I.F.B61742565',
    '9238-SD BERNARD0 DE LA T0RRE',
    'Te1éfono:928493638',
    'Centro Vend. Documento',
    'Fecha',
    'Hora',
    '9238 7868352024/923813-0006086615/04/2024 16:01',
    'ARTICULO',
    'IMPORTE',
    'FRESA TARRINA 500 GR',
    '1,59',
    'HIPERDINO ACEITUNA R/ANCHOA LATA 350',
    '0,95',
    'DESPERADOS CERVEZA TOQUE TEQUILA BOT',
    '1,05',
    'HIPERDINO CENTRO JAMON SERRANO BODEG',
    '0.310x13,62€/kg',
    '4,22',
    'MONTESANO JAMON COCIDO SELECCION KG',
    '0,308 x 8,74 €/kg',
    'Dto.0,54€',
    '2,15',
    'Total Articulos: 5',
    'TOTAL COMPRA:',
    '9,96',
    'Detalle de pagos',
    'EFECTIVO',
    '0,00',
    'TARJETA CREDITO',
    '9,96',
    'EMPLEAD0:12789.TICKET_P.E.203659',
    'HORA:160142',
    'FECHA-15/04/2024',
    'IMP0RTE9,96',
    'TARJETAxxxxxxxx*xxx5597',
    '087663',
    'CAPTURA CHIP / AUTORIZACION:',
    'LABEL: Mastercard',
    'ARC: 00',
    'ATC:004F',
    'AID:A0000000041010',
    'AUTENTICACION: Contact1ess EMV',
    'DCC INTERNACIONAL/REDSYS PCI',
    'COM. PE: 154197156',
    'TER. PE: 00000001',
    'SES. PE:15042024001'
]

receipt_texts_2 = [
    'SPAR TAFIRA',
    'C/.BRUNO NARANJO DIAZ9A-B',
    'TLF.:928351616-FAX:928351004',
    'NIFB02868248',
    'SUPERMERCAD0S DABEL2021,S.L',
    'TAFIRA BAJA',
    'FACTURA SIMPLIFICADA',
    'Nro.014002-18965',
    'Fecha:06-04-202415:23',
    'Cajerc:10074',
    'CANT.',
    'PVP IMPORTE',
    'DESCRIPCION',
    '1,49',
    '1,49',
    'CLIPPER MANZ.1.5L.',
    '1',
    '1,40',
    '1,99',
    'PLATANO PRIMERA GRAN',
    '2,79',
    '2,99',
    '2.99',
    'MANZANA PINK LADY GR',
    '3,10',
    '3,10',
    'SALSA.BARI.PES.GEN.1',
    '1,85',
    '1,85',
    'GOFIO B.LUGAR MIL.FU',
    '1',
    '1,75',
    '1,75',
    'ZUM.DISF.D.SIMON PIN',
    '1',
    '1,15',
    '1,15',
    'LECHE.GRNJ.FLR.UHT.',
    '1',
    'Lineas : 7',
    'Total F',
    '15,12',
    '"TARJETA',
    '15.12',
    'Entregado',
    'Cambio',
    '0,00',
    'Operacion',
    ': VENTA',
    '06/04/202415:24',
    'Fecha',
    'Comercio',
    '249060518',
    'ARC',
    '00',
    'A0000000031010',
    'AID',
    'Visa DEBIT',
    'App Labe1',
    '************761',
    'Tarjeta',
    '15,12EUR',
    'Importe',
    '-Copia para al'
]

receipt_texts_4 = [
    'S.A.',
    'MERCADONA.',
    'A-46103834',
    'AVDA. PINTOR FELO MONZON (C.C. 7 PALMAS)',
    'S/N',
    '35019 LAS PALMAS DE GRAN CANARIA',
    '928411755',
    'TELEFONO:',
    '03/04/202421:220P:144041',
    'FACTURA SIMPLIFICADA:2185-013-6970Z2',
    'Imp.)',
    'P.Unit',
    'Descripción',
    '3,30',
    '1 DETERG HIPO COLONIA',
    '13,50',
    '4,50',
    '3 SOLOMILLO POLLO CONG',
    '2,32',
    '1 JAMONCITO BARBACOA',
    '2,76',
    '1 JAMONCITO BARBACOA',
    '2,00',
    '1 NUEZ NATURAL',
    '1,25',
    '2,50',
    '2 QUESU COTIAGE',
    '6,52',
    '1 POLLO ENTERO LIMPIO',
    '1,70',
    '1 PAPEL VEGETAL 30H',
    '1.30',
    '1 BEBIDA AVELLANAS',
    '1,05',
    '1 INFUSION DORMIR',
    '1,40',
    '1 LECHE DE COCO',
    '1,35',
    '1 QUESO UNTAR LIGHT',
    '1 RULITO CABRA',
    '2,45',
    '1 GRIEGO LIGERO',
    '1,65',
    '1 BOLSA PLASTICO',
    '0,15',
    'TOTAL @)',
    '43,95',
    'TARJETA BANCARIA',
    '43,95',
    'COMERCIANTE MINORISTA',
    'TARJBANCARIA',
    '******915',
    'N.C072850332',
    'AUT:1LPOXG',
    'AIDA0000000041010',
    'ARC:3030',
    ')',
    'Importe43,95',
    'DEBIT MASTERCARD'
]


######################################################################
###  SCHEMA ###
######################################################################

class ProductCategory(str, Enum):
    fruits = 'fruits'
    vegetables = 'vegetables'
    protein_foods = 'protein_foods'
    seafood = 'seafood'
    dairy = 'dairy'
    grains = 'grains'
    nuts_and_seeds = 'nuts_and_seeds'
    sweets = 'sweets'
    spices = 'spices'
    beverages = 'beverages'
    snacks = 'snacks'
    condiments = 'condiments'
    frozen_foods = 'frozen_foods'
    bakery = 'bakery'
    canned_goods = 'canned_goods'
    household = 'household'
    personal_care = 'personal_care'
    pet_supplies = 'pet_supplies'
    other = 'other'

class ItemInfo(BaseModel):
    name: str = Field(..., description="Name of the item")
    unit: float = Field(..., description="Quantity of the item")
    price: float = Field(..., description="Price per unit of the item")
    amount: float = Field(..., description="Total amount for the item")
    category: ProductCategory = Field(..., description="Category of the item")

class PaymentMethodEnum(str, Enum):
    tarjeta = 'tarjeta'
    efectivo = 'efectivo'

class ReceiptInfo(BaseModel):
    store: str = Field(..., description="Store name")
    address: str = Field(..., description="Address of the store")
    city: str = Field(..., description="City where the store is located")
    phone: str = Field(..., description="Phone number of the store")
    receipt_no: str = Field(..., description="Receipt number")
    date: str = Field(..., description="Date of the receipt in DD/MM/YYYY format")
    time: str = Field(..., description="Time of the transaction")
    items: List[ItemInfo] = Field(..., description="List of items purchased")
    total: float = Field(..., description="Total amount of the receipt")
    number_items: int = Field(..., description="Number of items in the receipt")
    payment_method: PaymentMethodEnum = Field(..., description="Payment method used")


######################################################################
### FUNCTIONS ###
######################################################################

# Directory to store user-specific data
DATA_STORAGE_DIR = 'user_data'

# Ensure the data storage directory exists
os.makedirs(DATA_STORAGE_DIR, exist_ok=True)

def get_model(api_key):

    model = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=api_key)
    structured_llm = model.with_structured_output(ReceiptInfo, method="json_mode")
    return  structured_llm

def get_user_data_file(session_id):
    return os.path.join(DATA_STORAGE_DIR, f'{session_id}.csv')

def ensure_user_data_file_exists(session_id):
    data_file = get_user_data_file(session_id)
    if not os.path.exists(data_file):
        empty_df = pd.DataFrame(columns=[
            'store', 'address', 'city', 'phone', 'receipt_no', 'date', 'time',
            'total', 'number_items', 'payment_method', 'week', 'month', 'name',
            'unit', 'price', 'amount', 'category'
        ])
        empty_df.to_csv(data_file, index=False)

def ensure_numeric_columns(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def ensure_category(df):
    categories = [
        'fruits', 'vegetables', 'protein_foods', 'seafood', 'dairy', 'grains', 'nuts_and_seeds',
        'sweets', 'spices', 'beverages', 'snacks', 'condiments', 'frozen_foods', 'bakery',
        'canned_goods', 'household', 'personal_care', 'pet_supplies', 'other'
    ]

    df['category'] = df['category'].apply(lambda x: x if x in categories else 'other')

    return df

def parse_dates(date_str):
    date_formats = [
        "%d/%m/%Y",  # Day/Month/Year
        "%Y-%m-%d",  # Year-Month-Day
        "%m/%d/%Y",  # Month/Day/Year
        "%d-%m-%Y",  # Day-Month-Year
        "%m-%d-%Y",  # Month-Day-Year
        "%Y/%m/%d",  # Year/Month/Day
        "%Y.%m.%d",  # Year.Month.Day
        "%d.%m.%Y",  # Day.Month.Year
        "%m.%d.%Y"   # Month.Day.Year
    ]

    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue

    return pd.to_datetime(date_str, errors='coerce')  

    

# def image_ocr(image):

#     img = cv2.imread(image)
#     # Perform OCR
#     paddleocr = PaddleOCR(lang="es",ocr_version="PP-OCRv4",show_log = False, use_gpu=True)

#     result = paddleocr.ocr(img, cls=True)
#     result = result[0]
#     text = [line[1][0] for line in result]
#     return  text

def image_ocr(image_files):
    paddleocr = PaddleOCR(lang="es", ocr_version="PP-OCRv4", show_log=False, use_gpu=True)
    all_texts = []
    for image_file in image_files:
        img = cv2.imread(image_file)
        result = paddleocr.ocr(img, cls=True)
        result = result[0]
        text = [line[1][0] for line in result]
        all_texts.append(" ".join(text))
    return all_texts

def structured_output(texts, api_key):
    # Initialize the model
    structured_llm = get_model(api_key)
    examples_cat = [
    {"input": f"{receipt_texts_1}", "output": f"{example_cat_1}"},
    {"input": f"{receipt_texts_2}", "output": f"{example_cat_2}"},
    {"input": f"{receipt_texts_4}", "output": f"{example_cat_4}"}
    ]

    example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt_cat = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples_cat,
    )

    few_shot_prompt_cat.format()

    system_message_cat = """You are POS receipt data expert, parse, detect, recognize and convert the receipt OCR image result into structure receipt data object.
    Next, assign a category to each item. Don't make up value not in the Input. Output must be a well-formed JSON object.```json
    """

    final_prompt_cat = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_message}"),
        few_shot_prompt_cat,
        ("human", "{input}"),
    ])

    chain = final_prompt_cat | structured_llm

    all_data = []

    for text in texts:
        data = chain.invoke({"system_message":system_message_cat,"input": text})
        all_data.append(data)

    # Transform the data to dataframe
    all_dfs = []

    for data in all_data:

        store_df = pd.DataFrame([{
            'store': data.get('store', None),
            'address': data.get('address', None),
            'city': data.get('city', None),
            'phone': data.get('phone', None),
            'receipt_no': data.get('receipt_no', None),
            #'date': pd.to_datetime(data.get('date', None), format='%d/%m/%Y'),
            'date': parse_dates(data.get('date', None)),
            'time': data.get('time', None),
            'total': pd.to_numeric(data.get('total', None), errors='coerce'),
            'number_items': pd.to_numeric(data.get('number_items', None), errors='coerce'),
            'payment_method': data.get('payment_method', None)
        }])

        #if 'date' in store_df.columns:
        store_df['date'] = pd.to_datetime(store_df['date'], errors='coerce')
        store_df['week'] = store_df['date'].dt.isocalendar().week
        store_df['month'] = store_df['date'].dt.strftime('%B')
        store_df['date'] = store_df['date'].dt.strftime('%d/%m/%Y')

        # Ensure the 'month' column is in the correct order
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        store_df['month'] = pd.Categorical(store_df['month'], categories=month_order, ordered=True)

        # Transform items data to dataframe
        items_df = pd.DataFrame(data.get('items', []))

        for column in ['unit', 'price', 'amount']:
            items_df[column] = pd.to_numeric(items_df[column], errors='coerce')

        items_df = ensure_category(items_df)

        # Concatenate store and items DataFrames

        df = pd.concat([store_df] * len(items_df), ignore_index=True)
        df = pd.concat([df, items_df], axis=1)
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)

# def update_combined_df(corrected_df, combined_df=None):
#     # If there's an existing combined DataFrame, concatenate new data
#     if combined_df is not None:
#         combined_df = pd.concat([combined_df, corrected_df], ignore_index=True)
#     else:
#         combined_df = corrected_df

#     return combined_df



def update_combined_df(new_data, session_id):
    # Ensure user data file exists
    ensure_user_data_file_exists(session_id)

    # Load the user's historical data
    data_file = get_user_data_file(session_id)
    combined_df = pd.read_csv(data_file)

    # Append the new data
    new_data =  new_data.dropna(how='all')
    combined_df = pd.concat([combined_df, new_data], ignore_index=True)

    # Ensure numeric columns and category column
    columns_to_ensure_numeric = ['total', 'number_items', 'unit', 'price', 'amount']
    combined_df = ensure_numeric_columns(combined_df, columns_to_ensure_numeric)
    combined_df = ensure_category(combined_df)

    # Save the updated combined data back to the user's file
    combined_df.to_csv(data_file, index=False)

    receipt_count = combined_df['receipt_no'].nunique()

    return combined_df, receipt_count


def image_to_df(image_files, api_key):

    text = image_ocr(image_files)
    # Convert OCR text to structured output
    df = structured_output(text, api_key)
    return df

def initialize_session():
    return str(uuid.uuid4())


######################################################################
###  VISUALIZATION FUNCTIONS ###
######################################################################

categories = [
            'protein_foods', 'dairy', 'fruits', 'vegetables', 'grains', 'nuts_and_seeds',
            'beverages', 'snacks', 'condiments', 'frozen_foods', 'bakery', 'canned_goods',
            'household', 'personal_care', 'pet_supplies', 'other', 'sweets', 'spices', 'seafood'
        ]

colors = [
            '#87c293','#6074ab','#6b9acf','#8bbde6','#aae0f3','#c8eded',
            '#d18b79','#dbac8c','#d18b79','#dbac8c','#e6cfa1','#e7ebbc',
            '#b2dba0','#70a18f ','#637c8f', '#949da8','#b56e75','#c98f8f', '#edd5ca'
        ]

color_map = {category: colors[i % len(colors)] for i, category in enumerate(categories)}

def visualize_expenses_vs_budget(combined_df, selected_month, budget=500):
    if combined_df is not None and not combined_df.empty:
        expenses_per_month = combined_df.groupby('month')['amount'].sum().reset_index()
        filtered_data = expenses_per_month[expenses_per_month['month'] == selected_month]
        total = filtered_data['amount'].sum()
        percent_budget_left = float(100 - (total / budget) * 100)

        labels = ['Expenses', 'Remaining Budget']
        values = [total, budget - total]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, hoverinfo='label+value', textinfo='none')])

        fig.update_traces(marker=dict(colors=['rgba(0, 0, 0, 0)', '#80ced6']), sort=False)

        fig.add_annotation(
            text=f'<b>{percent_budget_left:.0f}%<b>',
            x=0.5,
            y=0.53,
            showarrow=False,
            font=dict(size=35),
            align='center'
        )

        fig.add_annotation(
            text='of Budget left',
            x=0.5,
            y=0.4,
            showarrow=False,
            font=dict(size=14),
            align='center'
        )

        fig.update_layout(
            title=dict(text=f'<b>Expenses vs. Budget in {selected_month}</b>',
                      x=0.5,
                      font=dict(size=16, color='Grey', family='Arial, sans-serif')),

            showlegend=False,
            width=500,
            height=500
        )

        return fig
    else:
        return None

def visualize_budget_tracking(combined_df, expenses_per_month, budget=500):
    if combined_df is not None and not combined_df.empty:
        expenses_per_month = combined_df.groupby('month')['amount'].sum().reset_index()
        expenses_per_month['percentage_expenses'] = (expenses_per_month['amount'] / budget) * 100
        expenses_per_month['percentage_budget'] = 100 - expenses_per_month['percentage_expenses']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=expenses_per_month['month'],
            y=expenses_per_month['percentage_expenses'],
            name='Actual Expenses',
            marker_color='#eeac99',
            text=round(expenses_per_month['amount'], 0),
            hovertemplate='<b>%{y:.0f}%<br>Expenses: %{text} EUR<b>',
            textposition='inside',
            insidetextanchor='middle',
            textfont_size=14
        ))
        fig.add_trace(go.Bar(
            x=expenses_per_month['month'],
            y=expenses_per_month['percentage_budget'],
            name='Remaining Budget',
            marker_color='#80ced6',
            text=round(budget - round(expenses_per_month['amount']), 0),
            hovertemplate='<b>%{y:.0f}%<br>Remaining Budget: %{text} EUR<b>',
            textposition='inside',
            insidetextanchor='middle',
            textfont_size=14
        ))

        fig.update_layout(
            title=dict(text='<b>Budget Tracking: Expenses vs. Budget per Month</b>',
                      x=0.5,
                      font=dict(size=16, color='Grey', family='Arial, sans-serif')),
            xaxis_title='',
            yaxis=dict(title='Percentage', zeroline=False, showgrid=False),
            plot_bgcolor='rgb(242,242,242)',
            showlegend=True,
            legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2),
            barmode='relative',
            height=500,
            width=700
            #width=len(expenses_per_month['month'])*300
                  )

        return fig
    else:
        return None

def visualize_pie_chart(combined_df, selected_month):
    if combined_df is not None and not combined_df.empty:
        expenses_per_month_category = combined_df.groupby(['month', 'category'])['amount'].sum().reset_index()
        filtered_data = expenses_per_month_category[expenses_per_month_category['month'] == selected_month]
        fig = px.pie(filtered_data, values='amount', names='category',
                    hole=0.4,
                    color='category',
                    #color_discrete_sequence=px.colors.qualitative.Light24,
                    color_discrete_map=color_map,
                    labels={'amount': 'Expenses', 'category': 'Category'},
                    width=500,
                    height=500,
                    )

        fig.update_traces(textinfo='percent',
                          insidetextorientation='radial',
                          textposition='inside',
                          hovertemplate="<b>Category: %{customdata}<br>Expenses: %{value} EUR<b>",
                          customdata=expenses_per_month_category['category'])

        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgb(242,242,242)',
            title=dict(text=f'<b>Category Expenses in {selected_month}</b>',
                      x=0.5,
                      font=dict(size=16, color='Grey', family='Arial, sans-serif'))
    )

        return fig
    else:
        return None

def visualize_category_expenses(combined_df):
    if combined_df is not None and not combined_df.empty:
        expenses_per_month_category = combined_df.groupby(['month', 'category'])['amount'].sum().reset_index()
        fig = px.bar(expenses_per_month_category, x='month', y='amount', color='category',
                  barmode='stack',
                  #color_discrete_sequence=px.colors.qualitative.Light24,
                  color_discrete_map=color_map,
                  labels={'amount': 'Expenses', 'category': 'Category'},
                  height=500,
                  width=700
                  #width=expenses_per_month_category['month'].nunique()*300
                    )

        fig.update_traces(hovertemplate="<b>Expenses: %{y} EUR<b>")

        fig.update_layout(
        xaxis=dict(showticklabels=True, title='', showgrid=False),
        yaxis=dict(zeroline=False, showgrid=False),
        plot_bgcolor='rgb(242,242,242)',
        title=dict(text='<b>Category expenses per month</b>', x=0.5, font=dict(size=16, color='Grey', family='Arial, sans-serif'))
    )
        return fig
    else:
        return None

def visualize_price_distribution(combined_df):
    if combined_df is not None and not combined_df.empty:
        fig = px.box(combined_df, x='category', y='price', color='category',
                    title='Price Distribution by Category',
                    #color_discrete_sequence=px.colors.qualitative.Light24,
                    color_discrete_map=color_map,
                    height=500,
                    width=500
                    )


        fig.update_layout(
                    xaxis=dict(showticklabels=False, title=''),
                    yaxis=dict(zeroline=False),
                    #paper_bgcolor='rgb(233,233,233)',
                    plot_bgcolor='rgb(242,242,242)',
                    showlegend=False,
                    title=dict(text='<b>Price Distribution by Category<b>',
                              x=0.5, font=dict(size=16, color='Grey', family='Arial, sans-serif'))
                          )

        return fig
    else:
        return None

def visualize_trend_expenses(combined_df):
    if combined_df is not None and not combined_df.empty:
        expenses_over_time_category = combined_df.groupby(['date', 'category'])['amount'].sum().reset_index()

        fig = px.line(expenses_over_time_category, x='date', y='amount', color='category',
                      labels={'amount': 'Expenses', 'category': 'Category', 'date': 'Date'},
                      #color_discrete_sequence=px.colors.qualitative.Light24,
                      color_discrete_map=color_map,
                      text=expenses_over_time_category['category'],
                      width=700,
                      height=500
                      )
        fig.update_traces(mode="markers+lines",
                          hovertemplate="<b>%{text}: <br>Expenses: %{y} EUR </br> %{x}")

        fig.update_layout(
                      xaxis=dict(showticklabels=True, title='', showgrid=False),
                      yaxis=dict(zeroline=False, showgrid=False),
                      plot_bgcolor='#f0efef',
                      title=dict(text='<b>Trends in expenses over time<b>',
                                x=0.5,
                                font=dict(size=16, color='Grey', family='Arial, sans-serif')),
                          )
        return fig
    else:
        return None

def visualize_all(combined_df, selected_month, budget):

    fig1 = visualize_expenses_vs_budget(combined_df, selected_month, budget)
    fig2 = visualize_budget_tracking(combined_df, selected_month, budget)
    fig3 = visualize_pie_chart(combined_df,selected_month)
    fig4 = visualize_category_expenses(combined_df)
    fig5 = visualize_price_distribution(combined_df)
    fig6 = visualize_trend_expenses(combined_df)
    return fig1, fig2, fig3, fig4, fig5, fig6