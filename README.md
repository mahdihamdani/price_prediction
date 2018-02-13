# price_prediction
This is a python app which predicts the product price given a description and meta information about the product

# Prerequisites
- Python >= 2.7
- sklearn
- pandas
- numpy
- category_encoders ()
- haversine

# Running Training
```
python train_price.py --dataset data/dataset.csv
```

#Running Prediction
```
python predict_price.py --product_name "Pizza Margherita" --product_description 'Tomatensauce' --menu_category 'Fit Pizza' --city_id 9 --latitude 52.521918 --longitude 13.413215 --cuisine_characteristics "Amerikanisch, Gesundes Essen" --dish_type_characteristics "Gemüse"
```

#Discussion:
- Bag of n-grams featues were extracted from string columns with free text: different n-gram sizes tested
- Features with list of strings were preprocessed to have indicators
- Diffeent categorical features were tested on the columns
- Different regression models were tested as well --> random forests regression is kept in the end

# TODO
- Feature selection, simpler models with same performance?
