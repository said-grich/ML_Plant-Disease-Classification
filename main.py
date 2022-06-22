import io
from dash import dcc
from dash import html
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import base64
import torch  # Pytorch module
import torch.nn as nn  # for creating  neural networks
import torch.nn.functional as F  # for functions for calculating loss
from PIL import Image
from torchvision import transforms

import numpy as np
from PIL import Image

imageUploaded = Image.open("assets/default.jpg").convert('RGB');
result = (False, "Please Select a Valid Photo")
classes = ['Apple___Apple_scab',
           'Apple___Black_rot',
           'Apple___Cedar_apple_rust',
           'Apple___healthy',
           'Blueberry___healthy',
           'Cherry_(including_sour)___Powdery_mildew',
           'Cherry_(including_sour)___healthy',
           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
           'Corn_(maize)___Common_rust_',
           'Corn_(maize)___Northern_Leaf_Blight',
           'Corn_(maize)___healthy',
           'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)',
           'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
           'Grape___healthy',
           'Orange___Haunglongbing_(Citrus_greening)',
           'Peach___Bacterial_spot',
           'Peach___healthy',
           'Pepper,_bell___Bacterial_spot',
           'Pepper,_bell___healthy',
           'Potato___Early_blight',
           'Potato___Late_blight',
           'Potato___healthy',
           'Raspberry___healthy',
           'Soybean___healthy',
           'Squash___Powdery_mildew',
           'Strawberry___Leaf_scorch',
           'Strawberry___healthy',
           'Tomato___Bacterial_spot',
           'Tomato___Early_blight',
           'Tomato___Late_blight',
           'Tomato___Leaf_Mold',
           'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite',
           'Tomato___Target_Spot',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
           'Tomato___Tomato_mosaic_virus',
           'Tomato___healthy'];


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# for loading in the device (GPU if available else CPU)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine loss
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}  # Combine accuracies

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim : 128 x 64 x 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, xb):  # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


def predict_image(img, model):
    """Converts image to array and return the predicted class
        with highest probability"""
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), torch.device('cpu'))
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return classes[preds[0].item()]


def pridiction(img, model):
    convert_tensor = transforms.ToTensor()
    img = convert_tensor(img)
    return predict_image(img, model);


def load_and_preprocess(image):
    image1 = Image.open(image)
    rgb = Image.new('RGB', image1.size)
    rgb.paste(image1)
    image = rgb
    test_image = image.resize((256, 256))
    return test_image


def np_array_normalise(test_image):
    np_image = np.array(test_image)
    final_image = np.expand_dims(np_image, 0)
    return final_image


tabs_styles = {
    'height': '44px',
    'align-items': 'center'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'margin-left': "10px",
    'height': '50px',
    'backgroundColor': '#ffff',
    'display': 'flex',
    'flex-direction': 'row',
    'justify-content': 'center',
    'align-items': 'center',

}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#76EAD7',
    'color': 'white',
    'padding': '6px',
    'margin-left': "10px",
    'display': 'flex',
    'flex-direction': 'row',
    'justify-content': 'center',
    'align-items': 'center',

}
model = to_device(ResNet9(3, len(classes)), torch.device("cpu"));
model.load_state_dict(torch.load("assets/plant-disease-model.pth", map_location=torch.device('cpu')))
model.eval()
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div((

    html.Div([
        html.Div([
            html.Div(id="main-header", children=[
                html.H1('Plant Disease Classification', id="page_title"),

            ])
        ], className="create_container1 four columns", id="title"),

    ], id="header", className="row flex-display", style={"margin-bottom": "25px"}),

    html.Div([
        html.Div([
            dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
                dcc.Tab(label='Data', value='tab-1', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='CNN Model', value='tab-2', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='ResNet', value='tab-3', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Prediction', value='tab-4', style=tab_style, selected_style=tab_selected_style)
            ]),
            html.Div(id='tabs-content-inline')
        ], className="create_container3 eight columns", ),
    ], className="row flex-display"),

    html.Div([
        html.Div([

            html.P('Select Country', className='fix_label',
                   style={'color': 'black', 'margin-top': '2px', 'display': 'None'}),
            dcc.Dropdown(id='select_countries',
                         multi=False,
                         clearable=True,
                         disabled=False,
                         style={'display': 'None'},
                         value='Switzerland',
                         placeholder='Select Countries',
                         options=[{'label': "c", 'value': "c"}], className='dcc_compon'),

        ], className="create_container3 four columns", style={'margin-bottom': '20px'}),
    ], className="row flex-display"),

    html.Div([
        html.Div([
            html.P('Select Chart Type', className='fix_label', style={'color': 'black', 'display': 'None'}),
            dcc.RadioItems(id='radio_items',
                           labelStyle={"display": "inline-block"},
                           options=[
                               {'label': 'Line chart', 'value': 'line'},
                               {'label': 'Donut chart', 'value': 'donut'},
                               {'label': 'Horizontal bar chart', 'value': 'horizontal'}],
                           value='line',
                           style={'text-align': 'center', 'color': 'black', 'display': 'None'}, className='dcc_compon'),

            dcc.Graph(id='multi_chart',
                      style={'display': 'None'},
                      config={'displayModeBar': 'hover'}),

        ], className="create_container3 six columns"),

    ], className="row flex-display"),

    html.Div([
        html.Div([

            html.P('Select Chart Type', className='fix_label', style={'color': 'black', 'display': 'None'}),
            dcc.RadioItems(id='radio_items1',
                           labelStyle={"display": "inline-block"},
                           options=[
                               {'label': 'Line chart', 'value': 'line'},
                               {'label': 'Donut chart', 'value': 'donut'},
                               {'label': 'Horizontal bar chart', 'value': 'horizontal'}],
                           value='line',
                           style={'text-align': 'center', 'color': 'black', 'display': 'None'}, className='dcc_compon'),

            # html.P('Select Country', className = 'fix_label', style = {'color': 'black', 'margin-top': '2px', 'display': 'None'}),
            # dcc.Dropdown(id = 'select_countries1',
            #              multi = False,
            #              clearable = True,
            #              disabled = False,
            #              style = {'display': 'None'},
            #              value = 'Switzerland',
            #              placeholder = 'Select Countries',
            #              options = [{'label': c, 'value': c}
            #                         for c in (income['Country'].unique())], className = 'dcc_compon'),

        ], className="create_container3 four columns", style={'margin-bottom': '20px'}),
    ], className="row flex-display"),

    html.Div([
        html.Div([

            dcc.Graph(id='multi_chart1',
                      style={'display': 'None'},
                      config={'displayModeBar': 'hover'}),
        ], className="create_container3 six columns"),
    ], className="row flex-display"),

), id="mainContainer", style={"display": "flex", "flex-direction": "column"})


@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Div([
                html.Div([

                    html.P('Select Country', className='fix_label', style={'color': 'black', 'margin-top': '2px'}),
                    dcc.Dropdown(id='select_countries',
                                 multi=False,
                                 clearable=True,
                                 disabled=False,
                                 style={'display': True},
                                 value='Switzerland',
                                 placeholder='Select Countries',
                                 options=[{'label': "test1", 'value': "test2"}
                                          ], className='dcc_compon'),

                ], className="create_container2 six columns", style={'margin-top': '20px'}),
            ], className="row flex-display"),

            html.Div([
                html.Div([
                    html.P('Select Chart Type', className='fix_label', style={'color': 'black'}),
                    dcc.RadioItems(id='radio_items',
                                   labelStyle={"display": "inline-block"},
                                   options=[
                                       {'label': 'Line chart', 'value': 'line'},
                                       {'label': 'Donut chart', 'value': 'donut'},
                                       {'label': 'Horizontal bar chart', 'value': 'horizontal'}],
                                   value='line',
                                   style={'text-align': 'center', 'color': 'black'}, className='dcc_compon'),

                    dcc.Graph(id='multi_chart',
                              config={'displayModeBar': 'hover'}),

                ], className="create_container2 ten columns", style={'margin-top': '10px'}),

            ], className="row flex-display"),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.Div([

                    html.P('Select Chart Type', className='fix_label', style={'color': 'black'}),
                    dcc.RadioItems(id='radio_items1',
                                   labelStyle={"display": "inline-block"},
                                   options=[
                                       {'label': 'Line chart', 'value': 'line'},
                                       {'label': 'Donut chart', 'value': 'donut'},
                                       {'label': 'Horizontal bar chart', 'value': 'horizontal'}],
                                   value='line',
                                   style={'text-align': 'center', 'color': 'black'}, className='dcc_compon'),

                    # html.P('Select Country', className = 'fix_label', style = {'color': 'black', 'margin-top': '2px'}),
                    # dcc.Dropdown(id = 'select_countries1',
                    #              multi = False,
                    #              clearable = True,
                    #              disabled = False,
                    #              style = {'display': True},
                    #              value = 'Switzerland',
                    #              placeholder = 'Select Countries',
                    #              options = [{'label': c, 'value': c}
                    #                         for c in (income['Country'].unique())], className = 'dcc_compon'),

                ], className="create_container2 six columns", style={'margin-top': '20px'}),
            ], className="row flex-display"),

            html.Div([
                html.Div([

                    dcc.Graph(id='multi_chart1',
                              config={'displayModeBar': 'hover'}),

                ], className="create_container2 ten columns", style={'margin-top': '10px'}),

            ], className="row flex-display"),
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Display content here in tab 3',
                    style={'text-align': 'center', 'margin-top': '100px', 'color': 'black'})
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.Div(
                className="box",
                children=[html.H1('Prediction of Plant Disease :',
                                  style={'text-align': 'center', 'margin-top': '100px', 'color': '#1F4068'})]
            ),
            html.Div(

                style={'padding': '0 120px'}, children=[
                    dcc.Upload(
                        id='upload-photo',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=False,
                    ),
                    html.Div(className="imageContainer",
                             children=[html.Img(id="img", src=imageUploaded, className="col-12 col-md-8")]),
                    html.Div(className="imageContainer", children=[
                        dbc.Button("Predict the Image", color="light", id="open", n_clicks=0, className="me-1")]),
                    dbc.Modal(

                        [
                            dbc.ModalHeader("Result Of prediction"),
                            dbc.ModalBody(html.Div(className="imageContainer", children=[html.H1(id="result")])),
                        ],
                        id="open-centered",
                        is_open=False,
                        centered=True,
                    ),

                ]),

        ])


@app.callback(
    Output(component_id='img', component_property='src'),
    Input(component_id='upload-photo', component_property='contents')
)
def prediction(image):
    global model;
    global imageUploaded;
    global result;
    encoded_image = image.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    bytes_image = io.BytesIO(decoded_image)
    image = Image.open(bytes_image).convert('RGB')
    imageUploaded = image
    re = pridiction(image, model)
    re=re.replace("___"," ")
    re=re.replace("_"," ")
    if (re):
        result = (True, re)
    else:
        result = (False, "chPhoto")
    # image = Image.open(io.BytesIO(base64_decoded))
    # image_np = np.array(image)
    return imageUploaded;


@app.callback(
    [Output(component_id='open-centered', component_property='is_open'),
     Output(component_id='result', component_property='children')
     ],
    [Input("open", "n_clicks")],
    [State("open-centered", "is_open")],
)
def openModal(n1, is_open):
    global result;

    if (result[0]) and (n1 ):
        return(not is_open,result[1])
    return (is_open,result[1])


if __name__ == "__main__":
    app.run_server(debug=True)
