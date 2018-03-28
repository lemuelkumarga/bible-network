## Package Requirements

### Mac/Ubuntu Operating System
- Other OS-es have not been tested, and may cause unexpected errors.

## Cloning the Repository

Cloning the repository is not as straightforward due to the presence of git submodules.

Please replicate the steps below in Terminal to ensure success.

``` sh
# Clone the repo as usual
git clone https://github.com/lemuelkumarga/bible-network

# Initialize submodule
cd bible-network
git submodule init
git submodule update

# When cloned, submodules are detached from the HEAD. We attempt to rectify this issue to prevent problems in git
cd shared
git checkout -b tmp
git checkout master
git merge tmp
git branch -d tmp

# Return to original folder if desired
cd ../../
```


# Constructing Social Networks in the Bible

### <i>Lemuel Kumarga</i>


## Problem Description

Our social circles are huge parts of our lives. They represent who we interact with, and how much we interact with them. With the digitization of communication and socialization, finding out this circle within each individual is an easier task than before. A simple glimpse into social networking sites such as Facebook and LinkedIn allows us to see who our friends are, whilst the frequency of digital communication can be used as a proxy of our closeness with them.

However, such information was not easily quantifiable in the pre-technology era. By attempting to model past lives using modern concepts, we could potentially gain further information about the past. For this project, we will use Natural Language Processing (NLP) concepts to <b>construct a social network for the bible, with the aim to depeen our understanding of the gospel.</b>


## Preliminaries

First load the necessary modules for this exercise.


```python
import sys
sys.path.append('shared/')
import defaults as _d

# Load All Main Modules
_d.load({"pd":"pandas",
         "math":"math",
         "cl":"collections",
         "np":"numpy",
         "sp":"scipy",
         "re":"re",
         "mpl":"matplotlib",
         "plotly":"plotly",
         "nltk":"nltk",
         "wordcloud":"wordcloud",
         "PIL":"PIL",
         "operator":"operator"},
         globals())

# Load All Submodules
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import plotly.offline as py
import plotly.graph_objs as py_go

# If you can't find the module, run nltk.download() in python
from nltk import sent_tokenize, word_tokenize

_d.stylize()
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>





<link href="shared/css/defaults.css" rel="stylesheet"><link href="../../shared/css/definitions.css" rel="stylesheet"><link href="../../shared/css/general.css" rel="stylesheet"><link href="shared/css/python.css" rel="stylesheet"><script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script><script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script><script src="shared/js/styles.js"></script><script src="shared/js/popover.js"></script>



We will also construct helper functions to be used later on.


```python
# -------------------------------------
# Genre-Related Functions
# -------------------------------------
def __get_genre_groups():
    global _genre_group
    if "_genre_group" not in globals():
        _genre_group = bible.groupby("Genre",sort=False)
    return _genre_group

def __get_genre_colors():
    global _genre_colors
    if "_genre_colors" not in globals():
        color_pal = _d.get_color("palette")(len(__get_genre_groups()))
        color_dict = dict()
        ind = 0
        for name, _ in __get_genre_groups():
            color_dict[name] = color_pal[ind]
            ind += 1
        _genre_colors = color_dict
    return _genre_colors

def __get_genre_legends(rev = True):
    global _genre_legends
    global _genre_legends_rev
    if "_genre_legends" not in globals():
        _genre_legends = [mpatches.Patch(color=_d.bg_color,label="Genre")]
        for name, group in __get_genre_groups():
            legend_text = name + " (" + group.index[0]
            if (len(group.index) > 1):
                legend_text += " - " + group.index[-1]
            legend_text += ")"
            _genre_legends.append(mpatches.Patch(color=__get_genre_colors()[name], label=legend_text))
        _genre_legends_rev = _genre_legends[:0:-1]
        _genre_legends_rev.insert(0,_genre_legends[0])
    
    if rev:
        return _genre_legends_rev
    else:
        return _genre_legends

# -------------------------------------
# Word-Cloud Related Functions
# -------------------------------------
def __word_cloud(input, fig_size = (20,10), image = None, colors = None):
    
    # Step 1: If there is an image specified, we need to create a mask
    mask = None
    if (image != None):
        mask = np.array(PIL.Image.open(image))
        if (colors == "image_colors"):
            colors = wordcloud.ImageColorGenerator(mask)
    
    # Step 2: Set up default colors
    def_colors = mpl.colors.ListedColormap(_d.get_color())
    
    # Step 3: Generate Word Cloud
    #https://stackoverflow.com/questions/43043437/wordcloud-python-with-generate-from-frequencies
    wc = wordcloud.WordCloud(height=fig_size[1]*100,
                             width=fig_size[0]*100,
                             background_color=_d.bg_color,
                             mask = mask,
                             colormap = def_colors,
                             color_func = colors).generate_from_frequencies(input)

    # Step 4: Plot Word Cloud
    plt.figure(figsize=fig_size)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")

__get_legend_separator = mpatches.Patch(color=_d.bg_color,label="")    
    
def __get_minmax_legends(input, title, key_format = "{:.2f}"):
    output = []
    output.append(mpatches.Patch(color=_d.bg_color,label=title))
    max_item = max(input.items(), key=operator.itemgetter(1))
    output.append(mlines.Line2D([0], [0], marker='o', color=_d.bg_color, label="Max: " + key_format.format(max_item[1]) + " - " + max_item[0],
                      markerfacecolor=_d.ltxt_color, markersize=20))
    min_item = min(input.items(), key=operator.itemgetter(1))
    output.append(mlines.Line2D([0], [0], marker='o', color=_d.bg_color, label="Min: " + key_format.format(min_item[1]) + " - " + min_item[0],
                      markerfacecolor=_d.ltxt_color, markersize=10))
    return output

__min_color = _d.pollute_color(_d.bg_color,_d.txt_color,0.4)
def __get_saturate_legends(title):
    output = []
    output.append(mpatches.Patch(color=_d.bg_color,label=title))
    output.append(mpatches.Patch(color=_d.get_color(0),label="Concentrated In 1 Genre"))
    output.append(mpatches.Patch(color=_d.pollute_color(__min_color,_d.get_color(0),0.3), label="Spread Out Across\nMultiple Genres"))
    return output
    
```

## Exploration

### Loading the Data

For this exercise, we will be using the bible <a data-toggle="popover" title="" data-content="A collection of texts" data-original-title="Corpus">corpus</a> from <a href="https://www.kaggle.com/oswinrh/bible/data" target="_blank">Kaggle.</a> The data will be stored in abbreviated book keys, with each book containing the following attributes:

* <span class="hl">Book Name</span>: Full name of the book
* <span class="hl">Testament</span>: New (NT) or old (OT)
* <span class="hl">Genre</span>: Genre of the book
* <span class="hl">Chapters</span>: Number of chapters
* <span class="hl">Verses</span>: Total number of verses
* <span class="hl">Text</span>: The actual text of the book




```python
# Get all book statistics
abb = pd.read_csv("data/key_abbreviations_english.csv")\
        .query('p == 1')[["a","b"]]\
        .rename(columns={"a" : "Key"})
ot_nt = pd.read_csv("data/key_english.csv")\
          .rename(columns={"n" : "Name", "t" : "Testament"})
genres = pd.read_csv("data/key_genre_english.csv")\
           .rename(columns={"n" : "Genre"})

# Load the main biblical text
bible = pd.read_csv("data/t_asv.csv")\
          .groupby("b", as_index=False)\
          .agg({"c": pd.Series.nunique, "v": "size", "t":" ".join})\
          .rename(columns={"c": "Chapters","v": "Verses","t": "Text"})
# Perform some cleaning
bible['Text'] = bible['Text'].apply(lambda t: re.sub("[`']","",t))

# Join the remaining book statistics
bible = bible.join(abb.set_index('b'), on='b')\
             .join(ot_nt.set_index('b'), on='b')\
             .join(genres.set_index('g'), on='g')\
             .drop(['b', 'g'], axis=1)\
             .set_index('Key')\
             [["Name","Testament","Genre","Chapters","Verses","Text"]]
            
# Show the first few lines
bible.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Testament</th>
      <th>Genre</th>
      <th>Chapters</th>
      <th>Verses</th>
      <th>Text</th>
    </tr>
    <tr>
      <th>Key</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gen</th>
      <td>Genesis</td>
      <td>OT</td>
      <td>Law</td>
      <td>50</td>
      <td>1533</td>
      <td>In the beginning God created the heavens and t...</td>
    </tr>
    <tr>
      <th>Exo</th>
      <td>Exodus</td>
      <td>OT</td>
      <td>Law</td>
      <td>40</td>
      <td>1213</td>
      <td>Now these are the names of the sons of Israel,...</td>
    </tr>
    <tr>
      <th>Lev</th>
      <td>Leviticus</td>
      <td>OT</td>
      <td>Law</td>
      <td>27</td>
      <td>859</td>
      <td>And Jehovah called unto Moses, and spake unto ...</td>
    </tr>
    <tr>
      <th>Num</th>
      <td>Numbers</td>
      <td>OT</td>
      <td>Law</td>
      <td>36</td>
      <td>1288</td>
      <td>And Jehovah spake unto Moses in the wilderness...</td>
    </tr>
    <tr>
      <th>Deut</th>
      <td>Deuteronomy</td>
      <td>OT</td>
      <td>Law</td>
      <td>34</td>
      <td>959</td>
      <td>These are the words which Moses spake unto all...</td>
    </tr>
  </tbody>
</table>
</div>




### About the Data

We will also derive some language statistics from each book, mainly:

* <span class="hl">Sentences</span>: Number of sentences in each book.
* <span class="hl">Words</span>: Number of words in each book.


```python
# Add Sentences and Words columns
bible["Sentences"] = pd.Series(0, index=bible.index)
bible["Words"] = pd.Series(0, index=bible.index)

# Save Tokens
sent_tokens = OrderedDict()
word_tokens = OrderedDict()

for i, r in bible[["Text"]].iterrows():
    txt = r.str.cat()
    sent_tokens[i] = sent_tokenize(txt)
    word_tokens[i] = word_tokenize(txt)
    bible.at[i,'Sentences'] = len(sent_tokens[i])
    # Remove Punctuation
    bible.at[i,'Words'] = len([w for w in word_tokens[i] if re.match('\w+',w)])

# Show
bible[["Name","Testament","Genre","Chapters","Verses","Sentences","Words"]].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Testament</th>
      <th>Genre</th>
      <th>Chapters</th>
      <th>Verses</th>
      <th>Sentences</th>
      <th>Words</th>
    </tr>
    <tr>
      <th>Key</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gen</th>
      <td>Genesis</td>
      <td>OT</td>
      <td>Law</td>
      <td>50</td>
      <td>1533</td>
      <td>1756</td>
      <td>38097</td>
    </tr>
    <tr>
      <th>Exo</th>
      <td>Exodus</td>
      <td>OT</td>
      <td>Law</td>
      <td>40</td>
      <td>1213</td>
      <td>1116</td>
      <td>32177</td>
    </tr>
    <tr>
      <th>Lev</th>
      <td>Leviticus</td>
      <td>OT</td>
      <td>Law</td>
      <td>27</td>
      <td>859</td>
      <td>664</td>
      <td>23830</td>
    </tr>
    <tr>
      <th>Num</th>
      <td>Numbers</td>
      <td>OT</td>
      <td>Law</td>
      <td>36</td>
      <td>1288</td>
      <td>996</td>
      <td>32034</td>
    </tr>
    <tr>
      <th>Deut</th>
      <td>Deuteronomy</td>
      <td>OT</td>
      <td>Law</td>
      <td>34</td>
      <td>959</td>
      <td>745</td>
      <td>27952</td>
    </tr>
  </tbody>
</table>
</div>



#### Book Length

One of the most intuitive ways to understand the books' uneven distribution is to assume that we are doing devotions of each chapter a day. Under such a scenario, we will have the following timeline:   


```python
plt.figure(figsize=(20,5))
    
# Create Plots
yticks = []
ylabels = []
x_progress = 0
x_length = sum(bible["Chapters"])
y_progress = 0
y_length = len(bible["Chapters"])               
for name, group in __get_genre_groups():

    row_ids = [ bible.index.get_loc(i) for i in group.index ]

    # Part 1: Bars When Genre Is Still Being Read
    length = 0
    # For each book in the genre
    for idx in row_ids:

        # If we are reading this book in the anniversary 
        if (math.floor((x_progress + length)/365) < math.floor((x_progress + length + bible["Chapters"][idx])/365)):
            yticks.append(idx + 1)
            ylabels.append("{} ({}%)".format(bible.index[idx],round(idx/y_length * 100)))

        plt.broken_barh([(x_progress + length, bible["Chapters"][idx])],
                        (y_progress, (idx + 1) - y_progress),
                        facecolors = __get_genre_colors()[name])
        length += bible["Chapters"][idx]
    
    
    # Part 2: Bars When Genre has Been Read
    plt.broken_barh([(x_progress + length, x_length - x_progress - length)],
                    (y_progress, max(row_ids) + 1 - y_progress), 
                    facecolors = __get_genre_colors()[name])
    
    x_progress += length
    y_progress = max(row_ids) + 1
    

# Add Titles and Grid
plt.title("Chapter Distribution by Book")
plt.grid(color=_d.fade_color(_d.ltxt_color,0.5), linestyle='dashed')

# Add X-Axis Details
plt.xlabel("Time Since Start")
xticks = [365, 2 * 365, 3 * 365 ,sum(bible["Chapters"])]
xlabels = [ "Year 1", "Year 2", "Year 3", "Year 3\nMonth 3" ]
plt.xticks(xticks, xlabels)
plt.xlim(0,x_length)

# Add Y-Axis Details
yticks.append(y_length)
ylabels.append("{} ({}%)".format(bible.index[-1],round(1 * 100)))
plt.ylabel("% of Books Completed")
plt.yticks(yticks, ylabels)
plt.ylim(0, y_length)

# Add Legends
plt.legend(handles=__get_genre_legends(), bbox_to_anchor=[1.3, 1.0])

plt.show()
```


![png](README_files/README_11_0.png)


By the 1st year, we will have only completed 18% of the books on the bible. If this is not discouraging enough, after a further year, we would still not have completed the Old Testament (<span class="yellow-text">Law</span> to <span class="red-text">Prophets</span>). However, upon reaching the New Testament (<span class="blue-text">Gospels</span> to <span class="green-text">Apocalyptic</span>), we could complete the whole set of books within 9 months. The Old Testament is at least 3 times longer than the New Testament!

####  Chapter Length

Assuming that the average human reads <a href="http://www.readingsoft.com/" target="_blank">200 words per minute</a>, we can also estimate how long it will take to read 1 chapter every day:


```python
bible["Minutes_p_Chapter"] = bible["Words"] / bible["Chapters"] / 200.
inputs = []

deg_incr = 360. / len(bible.index)
for name, group in __get_genre_groups():
    
    # Insert Legend Item
    inputs.append(
        py_go.Scatterpolar(
            r = [0, 0, 0, 0],
            theta = [0, 0, 0, 0],
            name = name,
            legendgroup = name,
            mode = 'none',
            fill = 'toself',
            fillcolor = __get_genre_colors()[name],
            showlegend = True
        )
    
    )    
    
    # Insert Each Book
    for key, val in group["Minutes_p_Chapter"].items():
        inputs.append(
            py_go.Scatterpolar(
                r = [0, val, val, 0],
                theta = [0,bible.index.get_loc(key)*deg_incr,(bible.index.get_loc(key)+1)*deg_incr,0],
                name = bible["Name"][key],
                legendgroup = name,
                mode = 'none',
                hoverinfo ='text',
                text=bible["Name"][key] + ": " + "{:.1f}".format(val) + " min",
                fill = 'toself',
                fillcolor = __get_genre_colors()[name],
                showlegend = False
            )
        )


layout = py_go.Layout(_d.py_layout)
layout["autosize"] = False
layout["width"] = 450
layout["height"] = 330
layout["margin"] = dict(t=50,l=0,r=0,b=20)
layout["title"] = "Minutes Required to Read a Chapter"
layout["polar"]["angularaxis"]["visible"]=False

fig = py_go.Figure(data=inputs, layout=layout)
py.iplot(fig)
```


<div id="20d26553-bb7c-4a34-aae0-b12df35761eb" style="height: 330px; width: 450px;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("20d26553-bb7c-4a34-aae0-b12df35761eb", [{"type": "scatterpolar", "r": [0, 0, 0, 0], "theta": [0, 0, 0, 0], "name": "Law", "legendgroup": "Law", "mode": "none", "fill": "toself", "fillcolor": "#e4b600", "showlegend": true}, {"type": "scatterpolar", "r": [0, 3.8097000000000003, 3.8097000000000003, 0], "theta": [0, 0.0, 5.454545454545454, 0], "name": "Genesis", "legendgroup": "Law", "mode": "none", "hoverinfo": "text", "text": "Genesis: 3.8 min", "fill": "toself", "fillcolor": "#e4b600", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.022125, 4.022125, 0], "theta": [0, 5.454545454545454, 10.909090909090908, 0], "name": "Exodus", "legendgroup": "Law", "mode": "none", "hoverinfo": "text", "text": "Exodus: 4.0 min", "fill": "toself", "fillcolor": "#e4b600", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.412962962962963, 4.412962962962963, 0], "theta": [0, 10.909090909090908, 16.363636363636363, 0], "name": "Leviticus", "legendgroup": "Law", "mode": "none", "hoverinfo": "text", "text": "Leviticus: 4.4 min", "fill": "toself", "fillcolor": "#e4b600", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.449166666666667, 4.449166666666667, 0], "theta": [0, 16.363636363636363, 21.818181818181817, 0], "name": "Numbers", "legendgroup": "Law", "mode": "none", "hoverinfo": "text", "text": "Numbers: 4.4 min", "fill": "toself", "fillcolor": "#e4b600", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.110588235294117, 4.110588235294117, 0], "theta": [0, 21.818181818181817, 27.27272727272727, 0], "name": "Deuteronomy", "legendgroup": "Law", "mode": "none", "hoverinfo": "text", "text": "Deuteronomy: 4.1 min", "fill": "toself", "fillcolor": "#e4b600", "showlegend": false}, {"type": "scatterpolar", "r": [0, 0, 0, 0], "theta": [0, 0, 0, 0], "name": "History", "legendgroup": "History", "mode": "none", "fill": "toself", "fillcolor": "#ec7436", "showlegend": true}, {"type": "scatterpolar", "r": [0, 3.892916666666667, 3.892916666666667, 0], "theta": [0, 27.27272727272727, 32.72727272727273, 0], "name": "Joshua", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "Joshua: 3.9 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.486190476190476, 4.486190476190476, 0], "theta": [0, 32.72727272727273, 38.18181818181818, 0], "name": "Judges", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "Judges: 4.5 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 3.15625, 3.15625, 0], "theta": [0, 38.18181818181818, 43.63636363636363, 0], "name": "Ruth", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "Ruth: 3.2 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.005483870967742, 4.005483870967742, 0], "theta": [0, 43.63636363636363, 49.090909090909086, 0], "name": "1 Samuel", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "1 Samuel: 4.0 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.287083333333333, 4.287083333333333, 0], "theta": [0, 49.090909090909086, 54.54545454545454, 0], "name": "2 Samuel", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "2 Samuel: 4.3 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 5.52, 5.52, 0], "theta": [0, 54.54545454545454, 59.99999999999999, 0], "name": "1 Kings", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "1 Kings: 5.5 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.6642, 4.6642, 0], "theta": [0, 59.99999999999999, 65.45454545454545, 0], "name": "2 Kings", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "2 Kings: 4.7 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 3.4732758620689657, 3.4732758620689657, 0], "theta": [0, 65.45454545454545, 70.9090909090909, 0], "name": "1 Chronicles", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "1 Chronicles: 3.5 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 3.5845833333333332, 3.5845833333333332, 0], "theta": [0, 70.9090909090909, 76.36363636363636, 0], "name": "2 Chronicles", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "2 Chronicles: 3.6 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 3.6885000000000003, 3.6885000000000003, 0], "theta": [0, 76.36363636363636, 81.81818181818181, 0], "name": "Ezra", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "Ezra: 3.7 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.032692307692308, 4.032692307692308, 0], "theta": [0, 81.81818181818181, 87.27272727272727, 0], "name": "Nehemiah", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "Nehemiah: 4.0 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.8510000000000004, 2.8510000000000004, 0], "theta": [0, 87.27272727272727, 92.72727272727272, 0], "name": "Esther", "legendgroup": "History", "mode": "none", "hoverinfo": "text", "text": "Esther: 2.9 min", "fill": "toself", "fillcolor": "#ec7436", "showlegend": false}, {"type": "scatterpolar", "r": [0, 0, 0, 0], "theta": [0, 0, 0, 0], "name": "Wisdom", "legendgroup": "Wisdom", "mode": "none", "fill": "toself", "fillcolor": "#b94628", "showlegend": true}, {"type": "scatterpolar", "r": [0, 2.176547619047619, 2.176547619047619, 0], "theta": [0, 92.72727272727272, 98.18181818181817, 0], "name": "Job", "legendgroup": "Wisdom", "mode": "none", "hoverinfo": "text", "text": "Job: 2.2 min", "fill": "toself", "fillcolor": "#b94628", "showlegend": false}, {"type": "scatterpolar", "r": [0, 1.4546999999999999, 1.4546999999999999, 0], "theta": [0, 98.18181818181817, 103.63636363636363, 0], "name": "Psalms", "legendgroup": "Wisdom", "mode": "none", "hoverinfo": "text", "text": "Psalms: 1.5 min", "fill": "toself", "fillcolor": "#b94628", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.439516129032258, 2.439516129032258, 0], "theta": [0, 103.63636363636363, 109.09090909090908, 0], "name": "Proverbs", "legendgroup": "Wisdom", "mode": "none", "hoverinfo": "text", "text": "Proverbs: 2.4 min", "fill": "toself", "fillcolor": "#b94628", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.335, 2.335, 0], "theta": [0, 109.09090909090908, 114.54545454545453, 0], "name": "Ecclesiastes", "legendgroup": "Wisdom", "mode": "none", "hoverinfo": "text", "text": "Ecclesiastes: 2.3 min", "fill": "toself", "fillcolor": "#b94628", "showlegend": false}, {"type": "scatterpolar", "r": [0, 1.67375, 1.67375, 0], "theta": [0, 114.54545454545453, 119.99999999999999, 0], "name": "Song of Solomon", "legendgroup": "Wisdom", "mode": "none", "hoverinfo": "text", "text": "Song of Solomon: 1.7 min", "fill": "toself", "fillcolor": "#b94628", "showlegend": false}, {"type": "scatterpolar", "r": [0, 0, 0, 0], "theta": [0, 0, 0, 0], "name": "Prophets", "legendgroup": "Prophets", "mode": "none", "fill": "toself", "fillcolor": "#81393c", "showlegend": true}, {"type": "scatterpolar", "r": [0, 2.7772727272727273, 2.7772727272727273, 0], "theta": [0, 119.99999999999999, 125.45454545454544, 0], "name": "Isaiah", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Isaiah: 2.8 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.06375, 4.06375, 0], "theta": [0, 125.45454545454544, 130.9090909090909, 0], "name": "Jeremiah", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Jeremiah: 4.1 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 3.424, 3.424, 0], "theta": [0, 130.9090909090909, 136.36363636363635, 0], "name": "Lamentations", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Lamentations: 3.4 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.0934375, 4.0934375, 0], "theta": [0, 136.36363636363635, 141.8181818181818, 0], "name": "Ezekiel", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Ezekiel: 4.1 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.86875, 4.86875, 0], "theta": [0, 141.8181818181818, 147.27272727272725, 0], "name": "Daniel", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Daniel: 4.9 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 1.8496428571428571, 1.8496428571428571, 0], "theta": [0, 147.27272727272725, 152.72727272727272, 0], "name": "Hosea", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Hosea: 1.8 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 3.3033333333333332, 3.3033333333333332, 0], "theta": [0, 152.72727272727272, 158.18181818181816, 0], "name": "Joel", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Joel: 3.3 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.312222222222222, 2.312222222222222, 0], "theta": [0, 158.18181818181816, 163.63636363636363, 0], "name": "Amos", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Amos: 2.3 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 3.235, 3.235, 0], "theta": [0, 163.63636363636363, 169.09090909090907, 0], "name": "Obadiah", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Obadiah: 3.2 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 1.6425, 1.6425, 0], "theta": [0, 169.09090909090907, 174.54545454545453, 0], "name": "Jonah", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Jonah: 1.6 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.225714285714286, 2.225714285714286, 0], "theta": [0, 174.54545454545453, 180.0, 0], "name": "Micah", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Micah: 2.2 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.0533333333333332, 2.0533333333333332, 0], "theta": [0, 180.0, 185.45454545454544, 0], "name": "Nahum", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Nahum: 2.1 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.4016666666666664, 2.4016666666666664, 0], "theta": [0, 185.45454545454544, 190.9090909090909, 0], "name": "Habakkuk", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Habakkuk: 2.4 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.663333333333333, 2.663333333333333, 0], "theta": [0, 190.9090909090909, 196.36363636363635, 0], "name": "Zephaniah", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Zephaniah: 2.7 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.7275, 2.7275, 0], "theta": [0, 196.36363636363635, 201.8181818181818, 0], "name": "Haggai", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Haggai: 2.7 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.2578571428571426, 2.2578571428571426, 0], "theta": [0, 201.8181818181818, 207.27272727272725, 0], "name": "Zechariah", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Zechariah: 2.3 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.16125, 2.16125, 0], "theta": [0, 207.27272727272725, 212.72727272727272, 0], "name": "Malachi", "legendgroup": "Prophets", "mode": "none", "hoverinfo": "text", "text": "Malachi: 2.2 min", "fill": "toself", "fillcolor": "#81393c", "showlegend": false}, {"type": "scatterpolar", "r": [0, 0, 0, 0], "theta": [0, 0, 0, 0], "name": "Gospels", "legendgroup": "Gospels", "mode": "none", "fill": "toself", "fillcolor": "#514f68", "showlegend": true}, {"type": "scatterpolar", "r": [0, 4.186964285714286, 4.186964285714286, 0], "theta": [0, 212.72727272727272, 218.18181818181816, 0], "name": "Matthew", "legendgroup": "Gospels", "mode": "none", "hoverinfo": "text", "text": "Matthew: 4.2 min", "fill": "toself", "fillcolor": "#514f68", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.6634375, 4.6634375, 0], "theta": [0, 218.18181818181816, 223.63636363636363, 0], "name": "Mark", "legendgroup": "Gospels", "mode": "none", "hoverinfo": "text", "text": "Mark: 4.7 min", "fill": "toself", "fillcolor": "#514f68", "showlegend": false}, {"type": "scatterpolar", "r": [0, 5.3525, 5.3525, 0], "theta": [0, 223.63636363636363, 229.09090909090907, 0], "name": "Luke", "legendgroup": "Gospels", "mode": "none", "hoverinfo": "text", "text": "Luke: 5.4 min", "fill": "toself", "fillcolor": "#514f68", "showlegend": false}, {"type": "scatterpolar", "r": [0, 4.536904761904762, 4.536904761904762, 0], "theta": [0, 229.09090909090907, 234.54545454545453, 0], "name": "John", "legendgroup": "Gospels", "mode": "none", "hoverinfo": "text", "text": "John: 4.5 min", "fill": "toself", "fillcolor": "#514f68", "showlegend": false}, {"type": "scatterpolar", "r": [0, 0, 0, 0], "theta": [0, 0, 0, 0], "name": "Acts", "legendgroup": "Acts", "mode": "none", "fill": "toself", "fillcolor": "#337388", "showlegend": true}, {"type": "scatterpolar", "r": [0, 4.3307142857142855, 4.3307142857142855, 0], "theta": [0, 234.54545454545453, 239.99999999999997, 0], "name": "Acts", "legendgroup": "Acts", "mode": "none", "hoverinfo": "text", "text": "Acts: 4.3 min", "fill": "toself", "fillcolor": "#337388", "showlegend": false}, {"type": "scatterpolar", "r": [0, 0, 0, 0], "theta": [0, 0, 0, 0], "name": "Epistles", "legendgroup": "Epistles", "mode": "none", "fill": "toself", "fillcolor": "#348385", "showlegend": true}, {"type": "scatterpolar", "r": [0, 2.9634375, 2.9634375, 0], "theta": [0, 239.99999999999997, 245.45454545454544, 0], "name": "Romans", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "Romans: 3.0 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.9409375, 2.9409375, 0], "theta": [0, 245.45454545454544, 250.90909090909088, 0], "name": "1 Corinthians", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "1 Corinthians: 2.9 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.3723076923076922, 2.3723076923076922, 0], "theta": [0, 250.90909090909088, 256.3636363636363, 0], "name": "2 Corinthians", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "2 Corinthians: 2.4 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.605, 2.605, 0], "theta": [0, 256.3636363636363, 261.8181818181818, 0], "name": "Galatians", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "Galatians: 2.6 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.5583333333333336, 2.5583333333333336, 0], "theta": [0, 261.8181818181818, 267.27272727272725, 0], "name": "Ephesians", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "Ephesians: 2.6 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.78875, 2.78875, 0], "theta": [0, 267.27272727272725, 272.7272727272727, 0], "name": "Philippians", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "Philippians: 2.8 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.5, 2.5, 0], "theta": [0, 272.7272727272727, 278.1818181818182, 0], "name": "Colossians", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "Colossians: 2.5 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 1.837, 1.837, 0], "theta": [0, 278.1818181818182, 283.6363636363636, 0], "name": "1 Thessalonians", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "1 Thessalonians: 1.8 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 1.75, 1.75, 0], "theta": [0, 283.6363636363636, 289.09090909090907, 0], "name": "2 Thessalonians", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "2 Thessalonians: 1.8 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 1.9016666666666666, 1.9016666666666666, 0], "theta": [0, 289.09090909090907, 294.5454545454545, 0], "name": "1 Timothy", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "1 Timothy: 1.9 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.00625, 2.00625, 0], "theta": [0, 294.5454545454545, 300.0, 0], "name": "2 Timothy", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "2 Timothy: 2.0 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 1.4983333333333335, 1.4983333333333335, 0], "theta": [0, 300.0, 305.45454545454544, 0], "name": "Titus", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "Titus: 1.5 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.19, 2.19, 0], "theta": [0, 305.45454545454544, 310.9090909090909, 0], "name": "Philemon", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "Philemon: 2.2 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.7053846153846153, 2.7053846153846153, 0], "theta": [0, 310.9090909090909, 316.3636363636363, 0], "name": "Hebrews", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "Hebrews: 2.7 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.3040000000000003, 2.3040000000000003, 0], "theta": [0, 316.3636363636363, 321.8181818181818, 0], "name": "James", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "James: 2.3 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.419, 2.419, 0], "theta": [0, 321.8181818181818, 327.27272727272725, 0], "name": "1 Peter", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "1 Peter: 2.4 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.57, 2.57, 0], "theta": [0, 327.27272727272725, 332.7272727272727, 0], "name": "2 Peter", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "2 Peter: 2.6 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 2.484, 2.484, 0], "theta": [0, 332.7272727272727, 338.18181818181813, 0], "name": "1 John", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "1 John: 2.5 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 1.495, 1.495, 0], "theta": [0, 338.18181818181813, 343.6363636363636, 0], "name": "2 John", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "2 John: 1.5 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 1.495, 1.495, 0], "theta": [0, 343.6363636363636, 349.09090909090907, 0], "name": "3 John", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "3 John: 1.5 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 3.145, 3.145, 0], "theta": [0, 349.09090909090907, 354.5454545454545, 0], "name": "Jude", "legendgroup": "Epistles", "mode": "none", "hoverinfo": "text", "text": "Jude: 3.1 min", "fill": "toself", "fillcolor": "#348385", "showlegend": false}, {"type": "scatterpolar", "r": [0, 0, 0, 0], "theta": [0, 0, 0, 0], "name": "Apocalyptic", "legendgroup": "Apocalyptic", "mode": "none", "fill": "toself", "fillcolor": "#34675c", "showlegend": true}, {"type": "scatterpolar", "r": [0, 2.7590909090909093, 2.7590909090909093, 0], "theta": [0, 354.5454545454545, 360.0, 0], "name": "Revelation", "legendgroup": "Apocalyptic", "mode": "none", "hoverinfo": "text", "text": "Revelation: 2.8 min", "fill": "toself", "fillcolor": "#34675c", "showlegend": false}], {"font": {"family": "Source Sans Pro", "color": "#424242"}, "margin": {"t": 50, "l": 0, "r": 0, "b": 20}, "paper_bgcolor": "#ffffff", "plot_bgcolor": "#ffffff", "xaxis": {"color": "#424242bf"}, "yaxis": {"color": "#424242bf"}, "polar": {"bgcolor": "#ffffff", "angularaxis": {"color": "#424242bf", "direction": "clockwise", "visible": false}, "radialaxis": {"color": "#424242bf"}}, "autosize": false, "width": 450, "height": 330, "title": "Minutes Required to Read a Chapter"}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


From the chart above, we conclude that chapter lengths across books are varied as well. For example, a chapter in <span class="hl orange-text">1 Kings</span> will take around 5.5 minutes to read, whilst a chapter in <span class="hl red-text">Psalms</span> will take around 1.5 minutes to read. 

### Preliminary Insights

After obtaining an overview of the bible as a text, we will proceed to investigate the occurences of various characters in the bible.

#### The Trinity

The first point of interest is how much God appears at different book in the bible:



```python
def find_occurence(regex):
    output = OrderedDict()
    for name, group in __get_genre_groups():
        l = [len(re.findall(regex,wt.str.cat())) for _, wt in group[["Text"]].iterrows()]
        output[name] = (len(l),sum(l)/len(l))
    return output

entityToSearch = OrderedDict([('God', 'God|Lord|GOD|LORD'),
                              ('Father','Jehovah|Father'),
                              ('Son','Jesus|Christ|Emmanuel'),
                              ('Spirit','Spirit')])

ind = 0
# Construct Plots for Each Entity
f, splt = plt.subplots(1,len(entityToSearch.items()), figsize=(20,5))
for title, regex in entityToSearch.items():
    occurences = find_occurence(regex)
    splt[ind].set_title(title)
    splt[ind].set_xticks([])
    splt[ind].set_yticks([])
    x = 0
    for n, v in occurences.items():
        splt[ind].bar([x + v[0]/2],
                      [v[1]],
                      color = __get_genre_colors()[n],
                      width = v[0])
        x += v[0]
    ind += 1

# Insert Legends
plt.legend(handles=__get_genre_legends(False), bbox_to_anchor = [2.2, 1.05])

plt.show()
```


![png](README_files/README_15_0.png)


Unsurprisingly, words associated with God the Father (Jehovah/Father) appears prominently in the Old Testament, while words associated with God the Son (Jesus/Christ) hits a high frequency in the Gospel narratives. Word counts of the Spirit appears the highest in Acts. This sequence is in line with the story of the Gospel, where the events first transcribed were between God the Father and His people, followed by Jesus Christ and his believers, and finally with the Holy Spirit and the church.

(Note: The limitation of such an approach is the failure to capture regular words meant to symbolize God. For example, words such as "Lamb" in Revelations correspond to Christ, but such symbols were excluded as they would introduce false positives.)

#### Major Characters

Using <a href="http://bibleblender.com/2014/biblical-lessons/biblical-history/complete-list-of-major-minor-characters-in-bible" target="_blank">external sources</a>, we can also obtain a list of the major characters in the bible. This list can then be used as a reference to detect names within the bible:


```python
# Characters obtained from http://bibleblender.com/2014/biblical-lessons/biblical-history/complete-list-of-major-minor-characters-in-bible
characters_regex = 'Adam|Seth|Enos|Kenan|Mahalalel|Jared|Enoch|Methuselah|Lamech|Noah|Shem|Adam|Cain|Enoch|Irad|Mehujael|Methusael|Lamech|Tubal-cain|Arpachshad|Shelah|Eber|Peleg|Reu|Serug|Nahor|Terah|Abraham|Isaac|Jacob|Judah|Perez|Hezron|Ram|Amminadab|Nahshon|Salmon|Boaz|Obed|Jesse|David|Abel|Kenan|Enoch|Noah |Abraham|Isaac|Jacob|Joseph|Sarah|Rebecca|Rachel|Leah|Moses|Aaron|Miriam|Eldad|Medad|Phinehas|Joshua|Deborah|Gideon|Eli|Elkanah|Hannah|Abigail|Samuel|Gad|Nathan|David|Solomon|Jeduthun|Ahijah|Elijah|Elisha|Shemaiah|Iddo|Hanani|Jehu|Micaiah|Jahaziel|Eliezer|Zechariah|Huldah|Isaiah|Jeremiah|Ezekiel|Daniel|Hosea|Joel|Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|Beor|Balaam|Job|Amoz|Beeri|Baruch|Agur|Uriah|Buzi|Mordecai|Esther|Oded|Azariah|Abimelech|Saul|Ish-boseth|David|Solomon|Jeroboam|Nadab|Baasha|Elah|Zimri|Tibni|Omri|Ahab|Ahaziah|Jehoram|Jehu|Jehoahaz|Jehoash|Jeroboam|Zechariah|Shallum|Menahem|Pekahiah|Pekah|Hoshea|Rehoboam|Abijam|Asa|Jehoshaphat|Jehoram|Ahaziah|Athaliah|Jehoash|Amaziah|Uzziah|Jotham|Ahaz|Hezekiah|Manasseh|Amon|Josiah|Jehoahaz|Jehoiakim|Jeconiah|Zedekiah|Simon|John|Aristobulus|Alexander|Hyrcanus|Aristobulus|Antigonus|Herod|Herod|Herod|Philip|Salome|Agrippa|Agrippa|Simon|Aaron|Eleazar|Eli|Phinehas|Asher|Benjamin|Dan|Gad|Issachar|Joseph|Ephraim|Manasseh|Judah|Levi|Naphtali|Reuben|Simeon|Zebulun|Jesus|Mary|Joseph|James|Jude|Joses|Simon|Peter|Andrew|James|John|Philip|Bartholomew|Thomas|Matthew|James|Judas|Simon|Judas|Matthias|Paul|Barnabas|James|Jude|Caiaphas|Annas|Zechariah|Agabus|Anna|Simeon|John|Apollos|Aquila|Dionysius|Epaphras|Joseph|Lazarus|Luke|Mark|Martha|Mary|Mary|Nicodemus|Onesimus|Philemon'
character_freq = []
for name, group in __get_genre_groups():
    names = [re.findall(characters_regex,wt.str.cat()) for _, wt in group[["Text"]].iterrows()]
    l = [(w,name) for l in names for w in l]
    character_freq.extend(l)

# The frequency of each character occurence by genre
character_freq = nltk.ConditionalFreqDist(character_freq)

# Create color functions to determine the genre most associated with the character
def color_func(word, font_size, position, orientation, **kwargs):
    most_common_genre = character_freq[word].most_common(1)[0][0]
    intensity = 1. * character_freq[word][most_common_genre] / sum(character_freq[word].values())
    return _d.pollute_color(__min_color, __get_genre_colors()[most_common_genre],intensity)

# Plot word cloud for each name
inputs = {}
for n, fd in character_freq.items():
    inputs[n] = sum(fd.values())
__word_cloud(inputs, colors=color_func)

# Titles
plt.title("Major Character Occurences")

# Legends
legend_cloud = list(__get_genre_legends(False))
legend_cloud.append(__get_legend_separator)
legend_cloud.extend(__get_saturate_legends("Concentration"))
legend_cloud.append(__get_legend_separator)
legend_cloud.extend(__get_minmax_legends(inputs, "Word Count","{:d}"))
plt.legend(handles=legend_cloud, bbox_to_anchor = [1.28, 1.])
plt.show()


```


![png](README_files/README_17_0.png)


Based on the list, we conclude that <span class="hl orange-text">David</span> appears the most in the bible. In addition, his appearances seem to be concentrated within the <span class="hl orange-text">History</span> genre. This is in stark-contrast to <span class="hl">Jesus</span>, whose name appeared across multiple genres (in particular across the New Testament).

(Note: One limitation of this approach is the assumption that there is a 1-1 mapping for most names to the person. To improve the accuracy of the results, we would need to perform some name disambiguation - i.e. Does Saul refer to King Saul or Paul aka Saul? Does John refer to the disciple or the Baptist?. Unfortunately, such analysis are not within the scope of this project.) 

## Preparation

In order to construct a social network, we first need to identify the relevant entities in the bible. One approach is to find a list of characters from external sources, and then using that list to identify the entity. However, this method is <span class="hl">unscalable</span>. To illustrate this, suppose that we would like to construct a similar network for "Oliver Twist". Then, we would need to find a list of names associated with the book. But what happens if we are not able to find such a list? The project would come to a dead end.

As such, to reduce reliance on external sources, we need to develop a more robust approach in identifying the relevant characters in the bible.

### Finding the Entities

Fortunately, we are able to capture names due to the nature of English linguistics. Names fall under the category of "Proper Nouns", which we can detect using <a href="https://en.wikipedia.org/wiki/Part-of-speech_tagging" target="_blank">Part-of-Speech (POS) tagging</a>:



```python
tagged_word_tokens = OrderedDict((n, nltk.tag.pos_tag(wt)) for n, wt in word_tokens.items())
# Extract Only Proper Nouns and Add Index
proper_noun_tokens = OrderedDict((n, [(i, w[0]) for i, w in enumerate(wt) if w[1] == "NNP"]) for n, wt in tagged_word_tokens.items())
# Print 100 Most Common Words
noun_freq = nltk.FreqDist(w for n,wt in proper_noun_tokens.items() for i, w in wt)
", ".join([n for n, v in noun_freq.most_common(50)])
```




    'Jehovah, God, Israel, Lord, David, O, Jesus, Judah, Jerusalem, Thou, Moses, Egypt, Behold, Christ, Saul, Jacob, Aaron, Spirit, Babylon, Solomon, Son, Father, Abraham, Joseph, Joshua, Pharaoh, Jordan, Levites, Go, Thy, Ye, Moab, Psalm, Benjamin, Ephraim, My, Holy, A, Paul, Jews, Peter, Yea, Zion, Manasseh, Samuel, Jeremiah, Joab, John, Hezekiah, Isaac'



Based on the above, we have captured a majority of names by tagging them under Proper Nouns. However, there are also some false positive words such as <span class="hl">O, Go, Thy, Ye</span> that have to be removed. It is also interesting to see entities other than people, for example, <span class="hl">Jerusalem, Babylon</span> represent countries. In the next section, we will determine how to handle each case.

### Managing the Cases

The first case to handle is the occurence of words which are not proper nouns (<span class="hl">O, Go, Thy, Ye</span>). To solve this, we simply need to exclude such words from consideration:



```python
false_npp = ['O','Thou','Behold','Go','Thy','Ye','My','A','Yea','Thus','Come',
             'Therefore','Wherefore','Be','So','Hear','ye','Psalm','Selah','Arise','Woe','King','Speak',
             'Almighty','Who','How','Chief','thy','Fear','Musician','Which','High','Take','Most',
             'Shall','Lo','Let','Praise','Make','Nay','Say','River','Art','Amen','South','Lest',
             'Bring','Oh','Remember','Did','Teacher','Sea','Whosoever','Do','Every','Unto','Know',
             'Are','Mine','See','Tell','Whoso','Gods','Wilt','Red','Holy']
# Extract Only Proper Nouns and Add Index
proper_noun_tokens = OrderedDict((n, [(i, w) for i, w in wt if w not in false_npp]) for n, wt in proper_noun_tokens.items())
# Print 100 Most Common Words after excluding False Proper Nouns
noun_freq = nltk.FreqDist(w for n,wt in proper_noun_tokens.items() for i, w in wt)
", ".join([n for n, v in noun_freq.most_common(50)])
```




    'Jehovah, God, Israel, Lord, David, Jesus, Judah, Jerusalem, Moses, Egypt, Christ, Saul, Jacob, Aaron, Spirit, Babylon, Solomon, Son, Father, Abraham, Joseph, Joshua, Pharaoh, Jordan, Levites, Moab, Benjamin, Ephraim, Paul, Jews, Peter, Zion, Manasseh, Samuel, Jeremiah, Joab, John, Hezekiah, Isaac, Assyria, Samaria, Jonathan, Ammon, Jehovahs, Absalom, Gentiles, Jeroboam, Gilead, Philistines, Elijah'



The second case to consider is non-human entities. Some examples of these are nations (<span class="hl">Jerusalem, Babylon</span>), locations (<span class="hl">Galilee</span>) and false idols (<span class="hl">Baal</span>). Since the relationships between non-human entities can also yield useful insights, we will not be excluding such words.

The third case is symbols referencing to an entity (<span class="hl">Lord, Father, Son</span>). We will be including such words as well for the same reason as the second case.

### The Entity Cloud

Using the <span class="hl">Proper Noun</span> approach, we can subsequently plot these entities into a word cloud:


```python
# The frequency of each character occurence by genre
character_freq = nltk.ConditionalFreqDist((w[1],bible["Genre"][n]) for n,wt in proper_noun_tokens.items() for w in wt)

# Create color functions to determine the genre most associated with the character
def color_func(word, font_size, position, orientation, **kwargs):
    most_common_genre = character_freq[word].most_common(1)[0][0]
    intensity = 1. * character_freq[word][most_common_genre] / sum(character_freq[word].values())
    return _d.pollute_color(__min_color, __get_genre_colors()[most_common_genre],intensity)

# Plot word cloud for each name
inputs = {}
for n, fd in character_freq.items():
    inputs[n] = sum(fd.values())
__word_cloud(inputs, colors=color_func)

# Titles
plt.title("Entities in the Bible")

# Legends
legend_cloud = list(__get_genre_legends(False))
legend_cloud.append(__get_legend_separator)
legend_cloud.extend(__get_saturate_legends("Concentration"))
legend_cloud.append(__get_legend_separator)
legend_cloud.extend(__get_minmax_legends(inputs, "Word Count","{:d}"))
plt.legend(handles=legend_cloud, bbox_to_anchor = [1.28, 1.])
plt.show()
```


![png](README_files/README_23_0.png)


As can be seen, we have now expanded the word cloud of major characters (<span class="orange-text">David</span>, <span class="purple-text">Jesus</span>) into a larger entity of names, nations, symbols (amongst others). There are also some interesting patterns starting to emerge. For once, the word <span class="purple-text hl">Jesus</span> is dispersed across multiple genres, while the word <span class="cyan-text hl">Christ</span> is concentrated within the Epistles!



## Constructing the Network

### Building Networks and Edges

### The Social Network

### Network Slices

## Summary of Results
