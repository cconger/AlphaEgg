import ctypes
import ctypes.util

class TesseractError(Exception):
  pass

class Tesseract(object):
  _lib = None
  _api = None

  class TessBaseAPI(ctypes._Pointer):
    _type_ = type('_TessBaseAPI', (ctypes.Structure,), {})
  
  @classmethod
  def setup_lib(cls, lib_path=None):
    if cls._lib is not None:
      return
    if lib_path is None:
      lib_path = "C:\Program Files\Tesseract-OCR\libtesseract-5.dll"
    cls._lib = lib = ctypes.CDLL(lib_path)

    lib.TessBaseAPICreate.restype = cls.TessBaseAPI

    lib.TessBaseAPIDelete.restype = None
    lib.TessBaseAPIDelete.argtypes = (cls.TessBaseAPI,)

    lib.TessBaseAPIInit3.argtypes = (cls.TessBaseAPI, ctypes.c_char_p, ctypes.c_char_p)

    lib.TessBaseAPISetImage.restype = None
    lib.TessBaseAPISetImage.argtypes = (cls.TessBaseAPI, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)

    lib.TessBaseAPISetVariable.argtypes = (cls.TessBaseAPI, ctypes.c_char_p, ctypes.c_char_p)

    lib.TessBaseAPIGetUTF8Text.restype = ctypes.c_char_p
    lib.TessBaseAPIGetUTF8Text.argtypes = ( cls.TessBaseAPI, )
  
  def __init__(self, language=b'eng', datapath=None, lib_path=None):
    if self._lib is None:
      self.setup_lib(lib_path)
    self._api = self._lib.TessBaseAPICreate()
    print("Initializing Tesseract DLL")
    if self._lib.TessBaseAPIInit3(self._api, datapath, language):
      print("Tesseract Initialization Failed")
      raise TesseractError('init failed')
    
  def __del__(self):
    if not self._lib or not self._api:
      return
    if not getattr(self, 'closed', False):
      self._lib.TessBaseAPIDelete(self._api)
      self.closed = True
  
  def _check_setup(self):
    if not self._lib:
      raise TesseractError('lib not configured')
    if not self._api:
      raise TesseractError('api not created')
  
  def set_image(self, imagedata, width, height, bytes_per_pixel, bytes_per_line=None):
    self._check_setup()
    if bytes_per_line is None:
      bytes_per_line = width * bytes_per_pixel
    self._lib.TessBaseAPISetImage(self._api, imagedata.ctypes.data_as(ctypes.c_char_p), width, height, bytes_per_pixel, bytes_per_line)
  
  def set_variable(self, key, val):
    self._check_setup()
    self._lib.TessBaseAPISetVariable(self._api, key, val)
  
  def get_utf8_text(self):
    self._check_setup()
    return self._lib.TessBaseAPIGetUTF8Text(self._api)
  
  def get_text(self):
    self._check_setup()
    result = self._lib.TessBaseAPIGetUTF8Text(self._api)
    if result:
      return result.decode('utf-8')