import unittest
from warden.common import (parseFiles, parse_xml_file, parse_json_file,
                           readPickle, loadConfig)
from warden.updateData import transform_datafiles_to_pickle
from pathlib import Path
from pandas import to_datetime


script_dir = Path(__file__).parent.resolve()
data_dir = script_dir/"test_data"
dataset_dir = script_dir/"mixed_data_set"


class test_file_parsing(unittest.TestCase):
    date1 = to_datetime('2025-08-14T14:18:05', utc=False, errors='coerce')
    gold_xml_report = [{'date': date1, 'measurement': 20, 'gitSHA': 'e43552c548', 'readable_path': 'Main timer', 'date_only': date1.date(), 'number_of_slashes': 0},
                   {'date': date1, 'measurement': 2.8, 'gitSHA': 'e43552c548', 'readable_path': 'Main timer|Generate matrix', 'date_only': date1.date(), 'number_of_slashes': 1},
                   {'date': date1, 'measurement': 0.2, 'gitSHA': 'e43552c548', 'readable_path': 'Main timer|Operation Op*x', 'date_only': date1.date(), 'number_of_slashes': 1},
                   {'date': date1, 'measurement': 15, 'gitSHA': 'e43552c548', 'readable_path': 'Main timer|Total Solve Time', 'date_only': date1.date(), 'number_of_slashes': 1},
                   {'date': date1, 'measurement': 7.5, 'gitSHA': 'e43552c548', 'readable_path': 'Main timer|Total Solve Time|Operation Op*x', 'date_only': date1.date(), 'number_of_slashes': 2},
                   {'date': date1, 'measurement': 7.5, 'gitSHA': 'e43552c548', 'readable_path': 'Main timer|Total Solve Time|Remainder', 'date_only': date1.date(), 'number_of_slashes': 2},
                   {'date': date1, 'measurement': 2, 'gitSHA': 'e43552c548', 'readable_path': 'Main timer|Remainder', 'date_only': date1.date(), 'number_of_slashes': 1}]

    date2 = to_datetime('2026-05-28T10:41:50+00:00', utc=False, errors='coerce').replace(tzinfo=None)
    gold_json_report = [{'date': date2, 'measurement': 0.00014961368213903744, 'gitSHA': '1d9649fd4', 'readable_path': 'KokkosBlas3_GEMM/m:1000/n:1000/k:1000/manual_time', 'date_only': date2.date(), 'number_of_slashes': 0},
                        {'date': date2, 'measurement': 0.0012214727895652176, 'gitSHA': '1d9649fd4', 'readable_path': 'KokkosBlas3_GEMM/m:1000/n:1000/k:1000/manual_time', 'date_only': date2.date(), 'number_of_slashes': 0},
                        {'date': date2, 'measurement': 0.0013659322159533078, 'gitSHA': '1d9649fd4', 'readable_path': 'KokkosBlas3_GEMM/m:1000/n:1000/k:1000/manual_time', 'date_only': date2.date(), 'number_of_slashes': 0},
                        {'date': date2, 'measurement': 0.001176069535117057, 'gitSHA': '1d9649fd4', 'readable_path': 'KokkosBlas3_GEMM/m:1000/n:1000/k:1000/manual_time', 'date_only': date2.date(), 'number_of_slashes': 0}]

    date3 = to_datetime('2026-09-27 08:00:12')
    gold_benchpark_report = [{'date': date3, 'measurement': 1.843276, 'gitSHA': '82cbde3c145ae0a271b36b7f8fb640d9dc2c269e', 'readable_path': 'tuolumne / amg2023 / +rocm caliper=mpi,time / gpumode=TPX / Avg time/rank / main', 'date_only': date3.date(), 'number_of_slashes': 0, 'has_children': False}]

    def test_parse_xml_file(self):
        parse_xml_file(data_dir/"non_existent.xml")
        parse_xml_file(data_dir/"empty.xml")
        parse_xml_file(data_dir/"bad_report.xml")
        rows = parse_xml_file(data_dir/"report.xml")
        self.assertEqual(len(rows), 7)
        for idx in range(7):
            self.assertEqual(rows[idx]['date'], self.gold_xml_report[idx]['date'])
            self.assertEqual(rows[idx]['measurement'], self.gold_xml_report[idx]['measurement'])
            self.assertEqual(rows[idx]['gitSHA'], self.gold_xml_report[idx]['gitSHA'])
            self.assertEqual(rows[idx]['readable_path'], self.gold_xml_report[idx]['readable_path'])
            self.assertEqual(rows[idx]['date_only'], self.gold_xml_report[idx]['date_only'])
            self.assertEqual(rows[idx]['number_of_slashes'], self.gold_xml_report[idx]['number_of_slashes'])
            assert(rows[idx]["date"].tzinfo is None)

    def test_parse_json_file(self):
        data_rows = parse_json_file(data_dir/"KokkosKernels_Blas3_gemm_benchmark_2026-05-28_T10-25-39.json", None)
        self.assertEqual(len(data_rows), 4)
        for idx in range(4):
            self.assertEqual(data_rows[idx]['date'], self.gold_json_report[idx]['date'])
            self.assertEqual(data_rows[idx]['measurement'], self.gold_json_report[idx]['measurement'])
            self.assertEqual(data_rows[idx]['gitSHA'], self.gold_json_report[idx]['gitSHA'])
            self.assertEqual(data_rows[idx]['readable_path'], self.gold_json_report[idx]['readable_path'])
            self.assertEqual(data_rows[idx]['date_only'], self.gold_json_report[idx]['date_only'])
            self.assertEqual(data_rows[idx]['number_of_slashes'], self.gold_json_report[idx]['number_of_slashes'])
            assert(data_rows[idx]["date"].tzinfo is None)

    def test_parse_benchpark_json_file(self):
        data_rows = parse_json_file(data_dir/"amg2023-rocm-caliper-mpi-time-7654321-2026-09-27.json", None)
        self.assertEqual(data_rows, self.gold_benchpark_report)
        assert(data_rows[0]["date"].tzinfo is None)

    def test_json_config(self):
        config = loadConfig(data_dir/"config_kokkoskernelsperf.yaml")

    def test_parseFiles(self):
        pickleFile = script_dir/"temp.pkl"
        pickleFile.unlink(missing_ok=True)
        parseFiles([], pickleFile, data_dir, None)
        pickleFile.exists()
        df = readPickle(pickleFile)
        self.assertEqual(len(df), 7+4+1)
        pickleFile.unlink(missing_ok=True)

class test_data_updates(unittest.TestCase):

    def test_transform_datafiles(self):
        config = loadConfig(dataset_dir/"config_test.yaml")
        baseDir = config["baseDir"]
        gitReposLocation = baseDir/config["gitRepositoriesLocation"]
        gitReposLocation.mkdir(exist_ok=True, parents=True)
        dataLocation = baseDir/config["dataLocation"]
        dataLocation.mkdir(exist_ok=True, parents=True)

        transform_datafiles_to_pickle(config, gitReposLocation, dataLocation)

        pickleFile = dataLocation/'Polaris.hkl'
        self.assertTrue(pickleFile.exists())
        df = readPickle(pickleFile)
        self.assertEqual(len(df), 5)

if __name__ == '__main__':
    unittest.main()
