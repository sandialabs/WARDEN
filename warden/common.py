import pandas as pd
import xml.etree.ElementTree as etree
import yaml
import json
import multiprocessing as mp
from datetime import timedelta, datetime
from itertools import chain
from pathlib import Path
from fnmatch import fnmatch
import logging

logger = logging.getLogger(__name__)


def updateWithDefaults(config, defaults):
    for k, v in defaults.items():
        if k in config:
            if isinstance(v, dict):
                updateWithDefaults(config[k], defaults[k])
        else:
            config[k] = v


def loadConfig(configFilename):
    '''
    Generate a default configuration and update it with config file values
    '''
    defaults = {"gitRepositoriesLocation": "GitRepos",
                "dataLocation": "ThePickleJar",
                "repositories": [],
                "annotations": {"filename": "MachineWideAnnotations.csv"},
                "server": {"host": "localhost",
                           "port": 8000},
                "code": {"name": "NAME_NOT_SET",
                         "url": "URL_NOT_SET"},
                "benchmark_files": [],
                "missingDataWarningAfterDays": 3,
                "debugMode": False}

    configFile = Path(configFilename)
    assert configFile.exists()
    config = None
    with open(configFile, "r") as cf:
        config = yaml.load(cf, Loader=yaml.Loader)
    updateWithDefaults(config, defaults)
    config["baseDir"] = configFile.parent.absolute()
    return config

def postprocess_data(config, data):
    """
    Update the data associated with a file and post-process it according to the config
    """
    top_rows = []
    if config is not None and 'benchmarks' in config:
        git_sha = data[0]['gitSHA']
        date_long = data[0]['date']
        date_short = data[0]['date_only']
        for config_bench in config['benchmarks']:
            top_timer = dict(name=config_bench['name'],
                             glob=config_bench['glob'] if 'glob' in config_bench else config_bench['name']+'*',
                             measurement=1.0,
                             num_children=0)
            for row in data:
                if fnmatch(row['readable_path'], top_timer['glob']):
                    row['readable_path'] = top_timer['name'] + '|' + row['readable_path']
                    row['number_of_slashes'] += 1
                    top_timer['num_children'] += 1
                    top_timer['measurement'] *= row['measurement']
            if 0 < int(top_timer['num_children']):
                data.append(dict(date=date_long,
                                 measurement=top_timer['measurement']**(1.0/top_timer['num_children']),
                                 gitSHA=git_sha,
                                 readable_path=top_timer['name'],
                                 date_only=date_short,
                                 number_of_slashes=0,
                                 has_children=True if 0 < top_timer['num_children'] else False))
            

    return data

# recursive traversal that carries the full path
def recurse(elem, path_components, date, git_sha, data_rows, deletePrefix):
    addedChild = False
    # only record nodes that actually have a value
    if 'name' in elem.attrib and 'value' in elem.attrib:
        name = elem.attrib['name']
        if deletePrefix is not None:
            name = name.removeprefix(deletePrefix)
        full_path = '|'.join(path_components + [name])
        data_row = {
            'date':         date,
            'measurement':  float(elem.attrib['value']),
            'gitSHA':       git_sha,
            'readable_path':full_path,
            'date_only':    date.date(),
            'number_of_slashes': full_path.count('|'),
        }
        data_rows.append(data_row)
        addedChild = True
    # recurse into children
    hasChildren = False
    for child in elem:
        hasChildren |= recurse(child, path_components + [name], date, git_sha, data_rows, deletePrefix)
    data_row["has_children"] = hasChildren
    return addedChild

def parse_xml_file(filename, skipBefore=None, deletePrefix=None):
    """
    Process the data of a single xml file
    """
    try:
        file_path = filename
        tree = etree.parse(file_path)
        root = tree.getroot()
        x_value = root.attrib['date']
        date = pd.to_datetime(x_value, utc=False, errors='coerce')
        if skipBefore is not None and date < skipBefore:
            return []
        meta = root.find('./metadata')
        git_sha = meta.attrib['value'] if meta is not None else 'No metadata found'

        # pick which top-level timing/memory branch you want
        base = root.find('./timing')
        if base is None:
            base = root.find('./memory')
        if base is None:
            return []

        # start recursion; path_components is empty so that the first element
        # produces "BaseTimer" (or whatever base.attrib['name'] is)
        data_rows = []
        recurse(base, [], date, git_sha, data_rows, deletePrefix)

        return data_rows
    except Exception as e:
        print(f"Could not parse tree because of the following error: {e}")
        return []

def parse_google_benchmark_json_data(json_data, skipBefore=None):
    """
    Process Google Benchmark json data.
    """
    date_long = None
    date_short = None
    git_sha = None
    if 'context' in json_data:
        if 'GIT_COMMIT_HASH' in json_data['context']:
            git_sha = json_data['context']['GIT_COMMIT_HASH']
        else:
            git_sha = 'No metadata found'
        if 'date' in json_data['context']:
            date_long = pd.to_datetime(json_data['context']['date'], errors='coerce')
            date_long = date_long.replace(tzinfo=None)
            date_short = date_long.date()

    data_rows = []
    if 'benchmarks' in json_data:
        for benchmark in json_data['benchmarks']:
            unit_scale = 1
            unit_to_scale = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns' : 1e-9}
            if 'time_unit' in benchmark:
                unit = benchmark['time_unit']
                if unit not in unit_to_scale:
                    emitWarning(
                        f"time_unit {unit} is not recognized!\n" +
                        f"Known units: {list(unit_to_scale.keys())}\n" +
                        f"Will skip timer {b['name']} using this unit.")
                    continue
                unit_scale = unit_to_scale[unit]
            time_seconds = float(benchmark['real_time']) * unit_scale
            data_rows.append(dict(date=date_long,
                                  measurement=time_seconds,
                                  gitSHA=git_sha,
                                  readable_path=benchmark['name'],
                                  date_only=date_short,
                                  number_of_slashes=0,
                                  has_children=False))
    return data_rows

def parse_benchpark_json_data(json_data, skipBefore=None):
    """
    Process Benchpark CI job metadata json data.
    """
    date_long = None
    date_short = None
    git_sha = 'No metadata found'
    if 'date' in json_data:
        date_long = pd.to_datetime(json_data['date'], errors='coerce')
        date_long = date_long.replace(tzinfo=None)
        date_short = date_long.date()
    if 'gitSHA' in json_data:
        git_sha = json_data['gitSHA']

    data_rows = []
    performance = json_data['performance']
    path_parts = [
        json_data['host'],
        json_data['benchmark'],
        json_data['variant'],
    ]
    if json_data['system_args']:
        path_parts.append(json_data['system_args'])
    path_parts += [performance['metric'], performance['region']]
    readable_path = ' / '.join(path_parts)

    data_rows.append(dict(date=date_long,
                          measurement=float(performance['value']) if performance['available'] else None,
                          gitSHA=git_sha,
                          readable_path=readable_path,
                          date_only=date_short,
                          number_of_slashes=0,
                          has_children=False,
                          status=json_data['status'],
                          failure_reason=performance.get('reason', '')))
    return data_rows

def parse_json_file(file_path, skipBefore=None):
    """
    Process the data of a single json file
    """
    json_data = None
    with open(file_path, "r") as fj:
        try:
            json_data = json.load(fj)
        except Exception as e:
            print(f"Could not parse json because of the following error: {e}")
            return []

    if 'benchmarks' in json_data:
        return parse_google_benchmark_json_data(json_data, skipBefore)
    if 'performance' in json_data:
        return parse_benchpark_json_data(json_data, skipBefore)

    logger.warning(f"Skipping {file_path}: unrecognized JSON format "
                   "(expected a 'benchmarks' or 'performance' key)")
    return []

def parse_file(file_path, benchmark_config, skipBefore=None, deletePrefix=None):
    """
    Dispatch the file to the appropriate reader and postprocess its data
    """
    data_rows = []
    if file_path.suffix == ".xml":
        data_rows = parse_xml_file(file_path, skipBefore, deletePrefix)
    elif file_path.suffix == ".json":
        data_rows = parse_json_file(file_path, skipBefore)

    if data_rows != []:
        # Check if the file we just parsed appears in our per file config
        benchmark_name = "_".join(file_path.stem.split("_")[:-2])
        file_config = None
        for bench_conf in benchmark_config:
            if 'filename' in bench_conf and bench_conf['filename'] == benchmark_name:
                data_rows = postprocess_data(bench_conf, data_rows)
                break

    return data_rows

def readPickle(pickleFilename):
    '''
    Read from a stored pickle
    '''
    # Load XML files and create DataFrame
    pickleFile = Path(pickleFilename)
    if pickleFile.exists():
        return pd.read_pickle(pickleFile, compression="xz")
    else:
        raise RuntimeError(f"The file {pickleFile} does not exists.")


def parseFiles(benchmark_config, pickleFilename, dataset_folder, maxNumDays=120, deletePrefix=None):
    """
    Parse a folder of XML/json files into a dataframe and store as pickle.
    """
    logger.info(f"Parsing files in {dataset_folder}")
    skipBefore = datetime.today()-timedelta(days=maxNumDays) if maxNumDays != None else None
    with mp.Pool() as pool:
        data_rows = pool.starmap(parse_file,
                                 map(lambda x: (x, benchmark_config, skipBefore, deletePrefix), Path(dataset_folder).glob("*")))

    logger.info(f"Mangling data for {dataset_folder}")
    df = pd.DataFrame(chain.from_iterable(data_rows))
    if len(df) > 0:

        df["date_only"] = pd.to_datetime(df["date_only"])

        df = df.sort_values(by='date')

        # sort so "previous" means earlier date
        df_sorted = df.sort_values(['readable_path', 'date_only'])

        # compute prevGitSHA = the gitSHA of the previous row within the same readable_path
        df_sorted['prevGitSHA'] = df_sorted.groupby('readable_path')['gitSHA'].shift(1)

        # put results back into original df (alignment by index)
        df['prevGitSHA'] = df_sorted['prevGitSHA']

        print(df.head(10))

        tmpPickleFilename = Path(Path(pickleFilename).parent/"temp.hkl")
        df.to_pickle(tmpPickleFilename, compression="xz")
        tmpPickleFilename.replace(pickleFilename)
    else:
        logger.info(f"After filtering by date, no files were read in {dataset_folder}. Not writing f{pickleFilename}.")
        pickleFilename.unlink(missing_ok=True)
