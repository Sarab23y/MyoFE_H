import json
import os
import tempfile
import unittest
import zipfile

from validation.export_ecm_corrections import FILES, build


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ECMCorrectionExportTests(unittest.TestCase):
    def test_bundle_is_allow_listed_and_auditable(self):
        handle, path = tempfile.mkstemp(suffix='.zip')
        os.close(handle)
        try:
            manifest = build(ROOT, path, 'daab60c', 'HEAD')
            self.assertEqual([entry['path'] for entry in manifest['files']],
                             list(FILES))
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                self.assertIn('TRANSFER_MANIFEST.json', names)
                self.assertIn('APPLY_INSTRUCTIONS.md', names)
                self.assertIn('ecm_corrections.patch', names)
                self.assertEqual(
                    set(name for name in names if name.startswith('files/')),
                    set('files/' + item for item in FILES))
                decoded = json.loads(
                    archive.read('TRANSFER_MANIFEST.json').decode('utf-8'))
                self.assertEqual(decoded['patch_baseline'],
                                 manifest['patch_baseline'])
                self.assertNotIn('data.csv', '\n'.join(names))
                self.assertFalse(any(name.endswith('.png') for name in names))
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
