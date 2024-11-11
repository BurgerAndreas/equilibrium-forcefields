# WandB custom charts with Vega-Lite

https://vega.github.io/vega-lite/

Standalone Vega-Lite editor: https://vega.github.io/editor/#/

## Data

### data selection
y: history, summary
x: config

### data transformation
indata(name, field, value)
Tests if the data set with a given name contains a datum with a field value that matches the input value. For example: indata('table', 'category', value)

### Combine data sources
"source": ["source1", "source2", ...]
{
  "name": "stats",
  "source": "table",
  "transform": [
    {
      "type": "aggregate",
      "groupby": ["x"],
      "ops": ["average", "sum", "min", "max"],
      "fields": ["y", "y", "y", "y"]
    }
  ]
}


https://vega.github.io/editor/?#/url/vega-lite/N4IgJghgLhDOCmVYgFykjAjKg2qKAlgLbyoAMANCAAoBO8ssArvapgL4X7GkqZV0GzVigBMnbiVSiB9Ri14BmdgF0qGCKNyTelEADEANgHsA7mwkhCUvlSNnpl67xkGT5lMpWdw0CKlAAOwgbXywQHyhaCEDYADNjWiJtEBNjAGsmAAdUKx4QKjjaY2S0MP8y4NCNLR90+ABPXOcCkDiCeEMwZBQcNwdvVSoiCFp03MMCQNIqeECAY2MwKYBzAJAAD3X2zrBcwXlWKigGrN4QAEcmGMIYQgA3Uh8msp2u3PtzY9Pzq5uCO4ER4RdjsIA
"transform": [
    {
      "lookup": "time",
      "from": {"data": {"name": "data2"}, "key": "time", "fields": ["Flow"]}
    }
  ],


### Combine two fields
To pull data into your chart from W&B, add template strings of the form "${field:<field-name>}" anywhere in your Vega spec. This will create a dropdown in the Chart Fields area on the right side, which users can use to select a query result column to map into Vega.

To set a default value for a field, use this syntax: "${field:<field-name>:<placeholder text>}"


{
  "calculate": "min(datum.${field:x1}, datum.${field:x2})",
  "as": "combined_x"
}

{
    "calculate": "datum.${field:x1} < datum.${field:x2} ? datum.${field:x1} : datum.${field:x2}",
    "as": "combined_x"
},

"calculate": "isValid(datum.${field:x1}) && isValid(datum.${field:x2}) ? min(datum.${field:x1}, datum.${field:x2}) : isValid(datum.${field:x1}) ? datum.${field:x1} : datum.${field:x2}",

### rename fields

"transform": [
{"calculate": "datum['Field With Space']", "as": "Field_With_Space"}
]

### select for DEQ

Does not work
{
      "calculate": "isValid(datum.model_is_deq) && model_is_deq == true ? 'DEQ' : isValid(datum.model_is_deq) && model_is_deq == false ? 'Equiformer' : 'default'",
      "as": "modeltype"
  }

Works
"transform": [
{
      "calculate": "isValid(datum.f_tol) ? 'DEQ' : 'Equiformer'",
      "as": "modeltype"
  }
]

### Legend

"color": {
"legend": {
        "labelExpr": "datum.label == 'tv' ? 'Tv' : datum.label == 'movie' ? 'Movie' :datum.label == 'video' ? 'Video' : 'Video Game'"
      },
}

ifTest ? thenValue : elseValue

"legend": {"title": null, "labelExpr": "datum.label == 'one' ? 'ONE' : datum.label == 'two' ? 'TWO' :datum.label == 'three' ? 'THREE' : 'DEFAULT'"}
"legend": {"title": null, "labelExpr": "datum.${field:colorstyle} == true ? 'DEQ' : datum.${field:colorstyle} == false ? 'Equiformer' : 'DEFAULT'"}

"legend": {"title": null, "labelExpr": "isBoolean(datum.${field:colorstyle}) ? 'bool' : isString(datum.${field:colorstyle}) ? 'str' : isDefined(datum.${field:colorstyle}) ? 'defined' : isValid(datum.${field:colorstyle}) ? 'valid' : 'DEFAULT'"}


      "encoding": {
        "x": {"field": "combined_x", "type": "quantitative", "axis": {"title": "Inference time [s]"}, "scale": {"domain": [0, 120]}},
        "y": {"field": "${field:y}", "type": "quantitative", "axis": {"title": r"Force MAE [kcal/mol/$\AA$]"}, "scale": {"domain": [0, 1]}},
        "color": {
          "field": "${field:colorstyle}",
          "scale": {"range": {"field": "color"}},
          "legend": {"title": "Model", "values": ["DEQ", "Equiformer"]}
        },
        "shape": {"field": "${field:markerstyle}", "type": "nominal", "legend": {"title": "Number of layers"}}
      }

### Legend: combine color and shape 




---

## Layout

### axis range and scaling

"scale": {"domain": [300,450]}
"scale": {"type": "log"}
"linear", "pow", "sqrt", "symlog", "log"

### plot types

https://vega.github.io/vega-lite/docs/line.html

"mark": "line",
"mark": {
    "type": "line",
    ...
},

"mark": {
    "type": "line",
    "point": true
},

### marker types

encoding the 'modeltype' field with the 'shape' encoding channel

https://vega.github.io/editor/#/url/vega-lite/N4KABGBEAkDODGALApgWwIaQFxUQFzwAdYsB6UgN2QHN0A6agSz0QFcAjOxge1IRQyUa6ALQAbZskoBmOgCtY3AHaQANOCgATZAgBOjQnh4qckAIJgE6Asl2Ex3PJcTcA7oyXUwL3bGSE3WzB0JU0wVEYxHTBCINoxByVYOjUNSE1rTBxgSFZdMWwtTNJ4dF95RRUAX3UISAxdAGtCnLwAT1jCyACPPDUoeG4HXS7dZE1IGrTkJUHND2oWjTqADxbIADNGZDEJ0wAJbl9-QJHVKHbO0wBHVhCjPGtGKknaiCg29a2dvagAWUiOgA+rFdECAOLoBLKfqQS7ILq3e7MJ4vKbvAZDI5fba7LoAeX0TBU5zhHQRpiU3AiSihr2WUFgiHQVzAOW+eNMhMYxNh8K6VJpdKqGhFIqAA

first:
"mark": {"type": "point"}

then in encoding:
"shape": {"field": "${field:markerstyle}", "type": "nominal"}

### Horizontal and vertical lines

    {
      "mark": {"type": "rule", "color": "gray", "strokeDash": [3, 3]},
      "encoding": {
        "y": {"value": 0.45}  // Adjust the y-value according to your preference
      }
    }

---

## Logging

tracemalloc
https://docs.python.org/3/library/tracemalloc.html

```python
import tracemalloc

tracemalloc.start()

print(tracemalloc.get_traced_memory())
 
tracemalloc.stop()
```