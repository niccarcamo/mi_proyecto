from django import forms

class MyPredictionForm(forms.Form):
    # Define los campos del formulario aquí
    campo1 = forms.CharField(max_length=100)
    campo2 = forms.IntegerField()
    # Añade más campos según sea necesario
