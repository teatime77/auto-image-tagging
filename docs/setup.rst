セットアップ
======================

.. code-block:: bash
    
    pip install tqdm

.. code-block:: bash

    pip install sphinxcontrib-plantuml


.. code-block:: python

    extensions = [
        'sphinxcontrib.plantuml',
    ]    

    plantuml = 'java -jar /usr/lib/plantuml.jar'


.. uml::

    Alice -> Bob: Hi!
    Alice <- Bob: How are you?

.. uml::

    object "inbound message" as m1
    object "XML Splitter" as s1

    m1 : <img:img/1.jpg>
    s1 : <img:img/2.jpg>
    m2 : <img:img/3.jpg>

    m1 -> s1
    s1 -> m2