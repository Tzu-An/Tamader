"""An asynchronous MP Quart server for Tamader"""
import os, sys
import psutil
import logging
import traceback
import json
import multiprocessing
import asyncio

from quart import Quart, request
from hypercorn.config import Config
from hypercorn.asyncio import serve
from cognite.processpool import ProcessPool

from scripts.tamader import Tamader
import scripts.common as common

multiprocessing.set_start_method('fork', force=True) # Windows bye!

app = Quart(__name__)


class Worker:
    def __init__(self):
        logger.warning("worker ready.")

    def run(self, inp, *args):
        """Get the output from Tamader calculater"""
        ret = {
            "success": False,
            "payload": inp,
            "output": [],
            "error": {}
        }

        try:
            output = agent.process(**inp)
            if output is not None:
                ret['success'] = True
                ret['output'] = output.tolist()

        except Exception as err:
            tb = traceback.format_exc()
            logger.error(f"Input: {inp}, Error: {err}")
            logger.error(tb)
            ret["error"] = {
                    "msg": str(err),
                    "traceback": tb
                }
        return ret
            
@app.route("/health")
async def health():
    """Heat beat"""
    return {"status": "UP"}

def get_memory_info(process):
    """Get memory info of a process"""
    mem = process.memory_info()
    output = {
        "rss": common.get_mb(mem.rss),
        "vms": common.get_mb(mem.vms),
    }
    return output

@app.route("/memory")
async def memory():
    """Get memory usage info"""
    process = psutil.Process(os.getpid())
    output = dict()
    output["main"] = get_memory_info(process)

    children = list()
    for child in process.children(True):
        children.append(get_memory_info(child))
    output["children"] = children

    return output

def wrap_future(future_obj):
    """Wrap external futures into a aysncio future object"""
    # due to asyncio.wrap_future deprecation
    loop = asyncio.get_event_loop()
    aio_future = loop.create_future()
    def on_done(*_):
        try:
            result = future_obj.result()
        except Exception as err:
            loop.call_soon_threadsafe(aio_future.set_exception, err)
        else:
            loop.call_soon_threadsafe(aio_future.set_result, result)
    future_obj.add_done_callback(on_done)
    return aio_future

@app.route("/calculate", methods=["POST"])
async def calculate():
    """Get the result from workers
       Only takes application/json type input
       with size smaller than max content length limit
       Return TimeoutError if timeout {"error": TimeoutError}
    """
    # Call request.data before getting json content to reject large post requests
    _ = await request.data
    inp = await request.get_json()

    future = pool.submit_job(inp)
    # <class 'concurrent.futures._base.Future'>
    try:
        ret = await asyncio.wait_for(wrap_future(future), 0.5)
    except Exception as err:
        tb = traceback.format_exc()
        logger.error("Exception: {}".format(err.__class__.__name__))
        logger.error(tb)
        ret = {"error": err.__class__.__name__}

    return json.dumps(ret)

def main():
    """Main function
        Set configurations, logger
        Create Tamader agent
        Create multiprocessing pool, bind port and run app
    """
    global pool, config, logger, agent
    config = common.get_config()
    logger = common.get_logger(__name__)
    common.set_logger_level(logger, config["logging_level"])
    agent = Tamader(logger=logger, max_retry=config.get("max_retry", 5))

    app.config['MAX_CONTENT_LENGTH'] = config.get("max_content_length", 2**10)

    pool = ProcessPool(Worker, config.get("rest_api_worker_number", 1))
    hypercorn_config = Config()
    hypercorn_config.bind = [":{0}".format(config.get("rest_api_port", 6543))]
    asyncio.run(serve(app, hypercorn_config))

if __name__ == '__main__':
    main()
