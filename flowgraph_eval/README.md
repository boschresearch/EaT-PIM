This directory includes code used to evaluate EaT-PIM's flow graph generation methods. 

For recreating results, you must obtain the flow graph data published by the [Yamakata Lab](https://sites.google.com/view/yy-lab/resource/english-recipe-flowgraph).
As the data is made available at request, it is not included in this repository.
This data can be acquired by following the contact instructions in the aforementioned page.

Once you obtain the full flow graph data from the Yamakata lab, you can place the 
data in the ```flowgraph_eval/data``` directory. This data should include r-100.json, r-200.json, and two directories (r-100 and r-200)
that contain a collection of recipe flowgraph data 
(e.g. ```flowgraph_eval/data/r-200/recipe-00000-05793.flow``` and ```flowgraph_eval/data/r-200/recipe-00000-05793.list```).

The evaluation can be run using the following scripts, where you can replace ```FG_EVAL``` with a name 
of your choosing if you want EaT-PIM's generated data in a different directory.

1. ```flowgraph_eval\modify_flowgraphs.py``` to transform the ground-truth flow graph data into a suitable form.

2. Conduct EaT-PIM's sentence parsing and flow graph generation
```eatpim\etl\parse_documents.py --output_dir FG_EVAL --input_file ../flowgraph_eval/flowgraph_recipes.csv```
```eatpim\etl\preprocess_unique_names_and_linking.py --input_dir FG_EVAL```
```eatpim\etl\transform_parse_results.py --input_dir FG_EVAL --n_cpu 2```
   
3. ```flowgraph_eval\compare_edges.py --data_path FG_EVAL``` run the evaluation
