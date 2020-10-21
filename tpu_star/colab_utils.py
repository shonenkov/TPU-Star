# -*- coding: utf-8 -*-

DRIVE_ROOT = '/content/drive/My Drive'


def get_notebook_name():
    """
    Get name of current Colab notebook.
    """
    import re
    import ipykernel
    import requests
    from requests.compat import urljoin
    from notebook.notebookapp import list_running_servers

    kernel_id = re.search('kernel-(.*).json', ipykernel.connect.get_connection_file()).group(1)
    servers = list_running_servers()
    for ss in servers:
        response = requests.get(urljoin(ss['url'], 'api/sessions'), params={'token': ss.get('token', '')})
        for nn in response.json():
            if nn['kernel']['id'] == kernel_id:
                notebook_name = nn['notebook']['name']
                return notebook_name
