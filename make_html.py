# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:00:58 2017

@author: ryuhei
"""

import os
import itertools
from glob import glob
from string import Template
import numpy as np


def make_html(images_dir_path='.'):
    html_template_str = '''
<html>
<head>
<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js"></script>
<script type="text/javascript" src="https://code.highcharts.com/highcharts.js"></script>
<script type="text/javascript" src="https://code.highcharts.com/modules/exporting.js"></script>
</head>

<body>
<div id="container"></div>

<script>
$(function () {
    Highcharts.chart('container', {
        chart: {
            type: 'scatter',
            width: 50000,
            height: 2000,
            zoomType: 'x',
            panning: true,
            panKey: 'shift',
            spacingBottom: 0
        },
        title: {
            text: 'Title'
        },
        xAxis: {
            title: {
                enabled: true,
                text: 'Error in logarithm of aspect ratio (error = truth - estimation, which means the more positive the wider)'
            },
            startOnTick: true,
            endOnTick: true,
            showLastLabel: true,
            tickInterval: 0.01,
        },
        yAxis: {
            minorGridLineWidth: 0
        },
        plotOptions: {
            scatter: {
                marker: {
                    radius: 5,
                    states: {
                        hover: {
                            enabled: true,
                            lineColor: 'rgb(100,100,100)'
                        }
                    }
                },
                states: {
                    hover: {
                        marker: {
                            enabled: false
                        }
                    }
                },
                tooltip: {
                    headerFormat: '',
                    pointFormat: '{point.x}'
                }
            }
        },
        series: [{
            name: 'Images',
            data: [${data}]
        }]
    });
});
</script>
</body>
</html>
'''

    glob_expr = os.path.join(images_dir_path, '*.png')
    data = []
    y_iter = itertools.cycle(range(200, 2000, 237))

    for filepath in glob(glob_expr):
        basename = os.path.basename(filepath)
        x = os.path.splitext(basename)[0]
#        y = np.random.uniform(-1000, 1000)
        y = next(y_iter)
        datum_template = Template('''
               {x: $x,
                y: $y,
                marker: {
                    symbol: 'url(./$file_name)'
                }}''')
        datum = datum_template.substitute(x=x, y=y, file_name=basename)
        data.append(datum)
    data_str = ",\n".join(data)

    html_template = Template(html_template_str)
    html_str = html_template.safe_substitute(data=data_str)

    output_file_path = os.path.join(images_dir_path, 'index.html')
    with open(output_file_path, 'w') as f:
        f.write(html_str)


if __name__ == '__main__':
    images_dir_path = '.'  # 画像が配置されているフォルダのパスを指定する
    make_html(images_dir_path)
