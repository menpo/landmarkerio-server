landmarkerio-server
===================
###The [Menpo](https://github.com/menpo/menpo)-powered [landmaker.io](https://github.com/menpo/landmarker.io) server

###About

Landmarker.io is a web app for annotating 2D and 3D assets. It has no
dependencies beyond on a modern browser, so it's trivial to set-up for
annotators. It expects to talk to a server that provides assets and annotations
over a simple RESTful API.

Menpo is a tool that makes loading a huge variety of 2D and 3D data trivial.

**landmarkerio-server** is an impliementation of the landmarker.io server API
in Python. It uses Menpo to load 2D and 3D assets, and serves them to
landmarker.io for annotation. When the annotator is done, it's
landmarkerio-server that will actually persist the landmarks to disk.

landmarkerio-server is ideal for quick annotation jobs on a local machine.
Once installed, just run the server (called `landmarkerio`) from the command
line. Your browser will automatically open to www.landmarker.io, and detect
the local server.

You can get as clever as you want to enable remote serving of landmarks -
SSH port tunnelling is an easy secure solution.

###Installation

landmarkerio-server requires [Menpo](https://github.com/menpo/menpo) to run. By
far the easiest way to install Menpo is via conda, see the Menpo wiki for
installation instructions for OS X, Linux and Windows.

Once you have Menpo, simply run

```
>> pip install landmarkerio-server
```

###Important concepts

landmarkerio-server handles three different forms of data

- assets - *meshes, textures and images*
- landmarks - *annotations on assets*
- templates - *specifications of landmarks*


###Usage

To begin annotating a folder of meshes, just run
```
>> landmarkerio ./path_to_mehes
```

You get help on the tool just as you would expect

```
>> landmarkerio --help
```
