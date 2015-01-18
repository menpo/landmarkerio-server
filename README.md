[![PyPI Release](http://img.shields.io/pypi/v/landmarkerio-server.svg?style=flat)](https://pypi.python.org/pypi/landmarkerio-server)

###The [Menpo](https://github.com/menpo/menpo)-powered [landmarker.io](https://github.com/menpo/landmarker.io) server

###About

Landmarker.io is a web app for annotating 2D and 3D assets. It has no
dependencies beyond on a modern browser, so it's trivial to set-up for
annotators. It expects to talk to a server that provides assets and annotations
over a simple RESTful API.

Menpo is a tool that makes loading a huge variety of 2D and 3D data trivial.

**landmarkerio server** is an implementation of the landmarker.io server API
in Python. It uses Menpo to load 2D and 3D assets, and serves them to
landmarker.io for annotation. When the annotator is done, it's
landmarkerio server that will actually persist the landmarks to disk.

The Python package for landmarkerio server is just landmarkerio.

landmarkerio is ideal for quick annotation jobs on a local machine.
Once installed, just run the server (called `lmio`) from the command
line. Your browser will automatically open to www.landmarker.io, and detect
the local server.

You can get as clever as you want to enable remote serving of landmarks -
SSH port tunnelling is an easy secure solution.

###Installation

landmarkerio server requires [Menpo](https://github.com/menpo/menpo)
[Menpo3d](https://github.com/menpo/menpo) to run. As these have somewhat
complex dependencies, by far the easiest way to install landmarkerio, is
with conda. On OS X, Linux or Windows, just install conda and then 
```
>> conda install -c menpo landmarkerio
```

###Important concepts

landmarkerio server handles three different forms of data

- assets - *meshes, textures and images*
- landmarks - *annotations on assets*
- templates - *specifications of landmarks*


###Usage

To begin annotating a folder of meshes, just run
```
>> lmio ./path_to_mehes
```

You get help on the tool just as you would expect

```
>> lmio --help
```

### Templates

#### Syntax

Templates restrict the set of allowed annotations and give the annotations
semantic meaning. The user of the server has full control over what
annotations the user of landmarker.io should complete by declaring *templates*.
A template file is simple a `.txt` file. The filename is the name of the template.
An example template is provided below.

**`face.txt`**
```text
mouth 6

nose 3
0 1
1 2

l_eye 8
0:7
7 0

r_eye 8
0:7
7 0

chin 1

```
We now annotate the file to explain the syntax:

**`face.txt`**
```
# This template is called 'face'. All saved landmarks
# will have the name 'face'
```

```text
mouth 6
# The first landmark group is called 'mouth'
# it is made up of six points

# A single line of whitespace seperates different
# groups from each other
nose 3
# The second group is 'nose' - 3 points.
0 1
1 2
# Pairs of numbers immediately following a declaration
# of a group specify connectivity information. Here,
# The first entry of the nose group is joined to the second
# (0-based indexing) and the second to the third. This will
# be visualized in the landmarker.

l_eye 8
0:7
# Slice notation is abused to construct straight chains
# of connectivity. This is expanded out into
# 0 1
# 1 2
# ...
# 6 7
7 0
# We can mix slicing with the pairings used above.
# Here we add a final connection back around from the 8th
# landmark to the first to complete the circle.

r_eye 8
0:7
7 0

chin 1
# Connectivity information is completely optional

```
The template format is intentionally extremely simple. The full
syntax is as displayed above. Note that there is no support for comments
so the second block is not a legal template.

#### Storing templates

A collection of template files can be placed in a templates folder.
A path to a folder can be provided as the `-t` argument to
`landmarkerio`. If no argument is provided, `lmio` looks for
the folder `~/.lmiotempates`. This provides a convenient place to
store frequently used templates.

Finally, it should be noted that landmarkerio currently doesn't support
switching templates (see
[landmarkerio#53](https://github.com/menpo/landmarker.io/issues/53)) and
as a result only the first template alphabetically is used for the time
being. This is only temporary.
