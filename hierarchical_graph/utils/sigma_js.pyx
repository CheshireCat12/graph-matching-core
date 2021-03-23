""" SigmaJS
@author: Anthony Gillioz

This script is used to create the graph from the core graph to the sigmaJS library.
The sigmaJS have to be put at the correct place to allow the created graphs to be drawn correctly.
"""
import json
import networkx as nx
import os
from string import Template
from pathlib import Path


def _round_3(num):
    return round(num, 3)

cdef class SigmaJS:
    """
    Class to transform the Graphs into html with the sigma.js library.

    Constants
    ---------
    _HTML_TEMPLATE : str
        Contains the html header with the paths to the SigmaJS lib.
        The js code is going to be filled in the HTML template
    _JS_TEMPLATE : str
        Contains the javascript code to create and draw the graphs

    Attributes
    ----------
    dataset: str
        The name of the dataset
    folder_results: str
        Folder where to write the results
    save_html: bool
       if to save the graph into the html format
    save_json: bool
       if to save the graph into the json format

    Methods
    -------


    """

    _HTML_TEMPLATE = Template('''
<!DOCTYPE html>
<html>
<head>
<title>Page Title</title>
</head>
<body>
<h1>$graph_name</h1>
<p>$extra_info_nodes</p>
<div id="graph-div" style="height:800px;border-style: solid;border-width: 1px;"></div>
<button id="layout" type="button">Layout</button>
<button id="export" type="export">Export</button>
</body>
<script src="$extra_level../../../external/sigma.js/src/sigma.core.js"></script>
<script src="$extra_level../../../external/sigma.js/src/conrad.js"></script>
<script src="$extra_level../../../external/sigma.js/src/utils/sigma.utils.js"></script>
<script src="$extra_level../../../external/sigma.js/src/utils/sigma.polyfills.js"></script>
<script src="$extra_level../../../external/sigma.js/src/sigma.settings.js"></script>
<script src="$extra_level../../../external/sigma.js/src/classes/sigma.classes.dispatcher.js"></script>
<script src="$extra_level../../../external/sigma.js/src/classes/sigma.classes.configurable.js"></script>
<script src="$extra_level../../../external/sigma.js/src/classes/sigma.classes.graph.js"></script>
<script src="$extra_level../../../external/sigma.js/src/classes/sigma.classes.camera.js"></script>
<script src="$extra_level../../../external/sigma.js/src/classes/sigma.classes.quad.js"></script>
<script src="$extra_level../../../external/sigma.js/src/classes/sigma.classes.edgequad.js"></script>
<script src="$extra_level../../../external/sigma.js/src/captors/sigma.captors.mouse.js"></script>
<script src="$extra_level../../../external/sigma.js/src/captors/sigma.captors.touch.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/sigma.renderers.canvas.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/sigma.renderers.webgl.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/sigma.renderers.svg.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/sigma.renderers.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/webgl/sigma.webgl.nodes.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/webgl/sigma.webgl.nodes.fast.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/webgl/sigma.webgl.edges.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/webgl/sigma.webgl.edges.fast.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/webgl/sigma.webgl.edges.arrow.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.labels.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.hovers.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.nodes.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.edges.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.edges.curve.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.edges.arrow.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.edges.curvedArrow.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.edgehovers.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.edgehovers.curve.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.edgehovers.arrow.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.edgehovers.curvedArrow.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/canvas/sigma.canvas.extremities.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/svg/sigma.svg.utils.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/svg/sigma.svg.nodes.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/svg/sigma.svg.edges.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/svg/sigma.svg.edges.curve.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/svg/sigma.svg.labels.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/renderers/svg/sigma.svg.hovers.def.js"></script>
<script src="$extra_level../../../external/sigma.js/src/middlewares/sigma.middlewares.rescale.js"></script>
<script src="$extra_level../../../external/sigma.js/src/middlewares/sigma.middlewares.copy.js"></script>
<script src="$extra_level../../../external/sigma.js/src/misc/sigma.misc.animation.js"></script>
<script src="$extra_level../../../external/sigma.js/src/misc/sigma.misc.bindEvents.js"></script>
<script src="$extra_level../../../external/sigma.js/src/misc/sigma.misc.bindDOMEvents.js"></script>
<script src="$extra_level../../../external/sigma.js/src/misc/sigma.misc.drawHovers.js"></script>
<script src="$extra_level../../../external/sigma.js/plugins/sigma.layout.forceAtlas2/worker.js"></script>
<script src="$extra_level../../../external/sigma.js/plugins/sigma.layout.forceAtlas2/supervisor.js"></script>
<script src="$extra_level../../../external/sigma.js/plugins/sigma.exporters.svg/sigma.exporters.svg.js"></script>
<script src="$extra_level../../../external/sigma.js/plugins/sigma.plugins.dragNodes/sigma.plugins.dragNodes.js"></script>
<script> $js_text </script>
</html>
''')

    _JS_TEMPLATE = Template('''

    var g = $graph_data ;

s = new sigma({graph: g, container: '$container', settings: { defaultNodeColor: '#ec5148', labelThreshold: $threshold, zoomingRatio:1.4} });

s.graph.nodes().forEach(function(n) {
  n.originalColor = n.color;
});
s.graph.edges().forEach(function(e) {
  e.originalColor = e.color;
});

s.bind('clickNode', function(e) {
  var nodeId = e.data.node.id,
      toKeep = s.graph.neighbors(nodeId);
  toKeep[nodeId] = e.data.node;

  s.graph.nodes().forEach(function(n) {
    if (toKeep[n.id])
      n.color = n.originalColor;
    else
      n.color = '#eee';
  });

  s.graph.edges().forEach(function(e) {
    if (toKeep[e.source] && toKeep[e.target])
      e.color = e.originalColor;
    else
      e.color = '#eee';
  });

  s.refresh();
});

s.bind('clickStage', function(e) {
  s.graph.nodes().forEach(function(n) {
    n.color = n.originalColor;
  });

  s.graph.edges().forEach(function(e) {
    e.color = e.originalColor;
  });

  s.refresh();
});

s.refresh();

// Listeners
var force = false;
document.getElementById('layout').onclick = function() {
  if (!force)
    s.startForceAtlas2({worker: true});
  else
    s.stopForceAtlas2();
  force = !force;
};

document.getElementById('export').onclick = function() {
  console.log('exporting...');
  var output = s.toSVG({download: true, filename: '$filename', size: 1000});
  // console.log(output);
};


// Initialize the dragNodes plugin:
var dragListener = sigma.plugins.dragNodes(s, s.renderers[0]);

dragListener.bind('startdrag', function(event) {
  console.log(event);
});
dragListener.bind('drag', function(event) {
  console.log(event);
});
dragListener.bind('drop', function(event) {
  console.log(event);
});
dragListener.bind('dragend', function(event) {
  console.log(event);
});


''')

    _THRESHOLDS = {'letter': 1,
                   'AIDS': 1,
                   'mutagenicity': 500,
                   'NCI1': 100,}

    def __init__(self,
                 str dataset,
                 str folder_results,
                 bint save_html=True,
                 bint save_json=False):
        self.dataset = dataset
        self.folder_results = folder_results
        self.save_html = save_html
        self.save_json = save_json

    def save_to_sigma_with_score(self,
                                 Graph graph,
                                 double[::1] centrality_score,
                                 str name_centrality_measure,
                                 int level=-1,
                                 str extra_info='',
                                 str extra_info_nodes='No info'):
        """
        Create and save the graph into the SigmaJS format.
        
        :param graph:
        :param centrality_score:
        :param name_centrality_measure:
        :param level:
        :param extra_info:
        :param extra_info_nodes:
        :return:
        """
        json_graph = self.graph_to_json_with_score(graph, centrality_score, name_centrality_measure)

        if self.save_json:
            self._write_to_file(json_graph,
                                graph.name,
                                name_centrality_measure,
                                extension='json',
                                level=level,
                                extra_info=extra_info)

        html_graph = self.graph_to_html(json_graph, graph.name, level, extra_info_nodes)

        if self.save_html:
            self._write_to_file(html_graph,
                                graph.name,
                                name_centrality_measure,
                                extension='html',
                                level=level,
                                extra_info=extra_info)

    def _write_to_file(self,
                       data,
                       str graph_name,
                       str centrality_measure,
                       str extension='html',
                       int level=-1,
                       str extra_info=''):
        """

        :param data:
        :param graph_name:
        :param centrality_measure:
        :param extension:
        :param level:
        :param extra_info:
        :return:
        """
        folder = self.folder_results
        prefix = ''
        suffix = ''

        if extra_info:
            prefix += f'{extra_info}_'

        if level >= 0:
            folder = os.path.join(folder, f'{centrality_measure}_{graph_name}', '')
            suffix = f'_{level}'


        Path(folder).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(folder,
                                f'{prefix}{centrality_measure}_{graph_name}{suffix}.{extension}')

        with open(filename, mode='w') as fp:
            if extension == 'json':
                json.dump(data, fp)
            elif extension == 'html':
                fp.write(data)
            else:
                raise ValueError(f'The extension {extension} is not accepted!')


    def graph_to_json_with_score(self, Graph graph,
                                 double[::1] centrality_score,
                                 str name_centrality_measure):
        """
        Convert the graph to the json format.
        Create a dict with the nodes and the edges of the given graph.
        The format of the nodes and edges correspond to the format of the SigmaJS lib.

        :param graph:
        :param centrality_score:
        :param name_centrality_measure:
        :return: The graph on a dict format.
        """
        cdef:
            Edge edge
            Node node

        graph_data = {'nodes': [], 'edges': []}

        for node, score in zip(graph.nodes, centrality_score):
            score = _round_3(score)
            lbl = f'{name_centrality_measure}: {score}; {node.label.sigma_attributes()}'
            x, y = [_round_3(val) for val in node.label.sigma_position()]
            graph_data['nodes'].append({
                'id': f'{node.idx}',
                'label': lbl,
                'x': x,
                'y': y,
                'size': score
            })

        for idx, edge in enumerate(graph._set_edge()):
            graph_data['edges'].append({
                'id': f'{idx}',
                'source': f'{edge.idx_node_start}',
                'target': f'{edge.idx_node_end}',
                'label': f'{edge.weight.valence}'
            })

        return graph_data

    def graph_to_html(self, dict graph, str graph_name, int level=-1, str extra_info_nodes='No info'):
        """
        Create the graph in SigmaJS format by filling the JS and HTML templates.

        :param graph:
        :param graph_name:
        :param level:
        :param extra_info_nodes:
        :return: The graph under html format
        """
        if self.dataset in ['mutagenicity', 'NCI1']:
            self._create_layout_mutagenicity(graph)

        prefix = f'{level}_' if level >= 0 else ''

        js_text = self._JS_TEMPLATE.substitute({'graph_data': json.dumps(graph),
                                                'container': 'graph-div',
                                                'threshold': self._THRESHOLDS[self.dataset],
                                                'filename': f'{graph_name}.svg',
                                               })
        html = self._HTML_TEMPLATE.substitute({'graph_name': f'{prefix}{graph_name}',
                                               'extra_info_nodes': extra_info_nodes,
                                               'extra_level': '../' * (level >= 0),
                                               'js_text': js_text})
        return html


    def _create_layout_mutagenicity(self, dict graph):
        """
        Create an extra layout for the graph.
        It is useful for the datasets that have not the coordinates of the nodes.
        Use the NetworkX lib to find a good approximation of position of the nodes.

        :param graph:
        :return:
        """
        nx_graph = nx.Graph()
        for node in graph['nodes']:
            nx_graph.add_node(node['id'])

        for edge in graph['edges']:
            nx_graph.add_edge(edge['source'], edge['target'], weight=1.)

        # pos = nx.drawing.layout.spring_layout(nx_graph, iterations=2500, center=[0.0, 0.0], seed=42)
        pos = nx.drawing.layout.kamada_kawai_layout(nx_graph)

        for node in graph['nodes']:
            node['x'], node['y'] = pos[node['id']]
