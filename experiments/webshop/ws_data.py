"""at WebShop Specifies the commodity and attribute files in the process（at WebShop of Python In the environment ws_worker.py / ws_convert.py call）。

equivalent to WebShop README described in“Load more commodities”modification method：Change web_agent_site/utils.py of DEFAULT_FILE_PATH and
DEFAULT_ATTR_PATH Two Path Constants。These two names are changed in the module when running here（engine.py at import Theyre incorrigible
its own namespace，So well change both places），Original file remains unchanged。See analysis plan.md 6.13 Revision 1。
"""
import os


def apply(webshop_dir, product_file, attr_file):
    import web_agent_site.engine.engine as E
    import web_agent_site.utils as U
    fp = os.path.join(webshop_dir, "data", product_file)
    ap = os.path.join(webshop_dir, "data", attr_file)
    for p in (fp, ap):
        if not os.path.exists(p):
            raise FileNotFoundError(f"not found WebShop data {p}（Run first bash webshop/setup_webshop.sh）")
    U.DEFAULT_FILE_PATH = fp
    U.DEFAULT_ATTR_PATH = ap
    E.DEFAULT_FILE_PATH = fp
    E.DEFAULT_ATTR_PATH = ap
    return fp, ap
