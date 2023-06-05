'use strict';

const {
  Array,
  FunctionPrototypeBind,
} = primordials;

const {
  // For easy access to the nextTick state in the C++ land,
  // and to avoid unnecessary calls into JS land.
  tickInfo,
  // Used to run V8's micro task queue.
  runMicro,
  setTickCallback,
  enqueueMicro
} = internalBinding('task_queue');

const {
  triggerUncaughtException
} = internalBinding('errors');

const {
  setHasRejectionToWarn,
  hasRejectionToWarn,
  listenForRejections,
  processPromiseRejections
} = require('internal/process/promises');

const {
  getDefaultTriggerAsyncId,
  newAsyncId,
  initHooksExist,
  destroyHooksExist,
  emitInit,
  emitBefore,
  emitAfter,
  emitDestroy,
  symbols: { async_id_symbol, trigger_async_id_symbol }
} = require('internal/async_hooks');
const {
  ERR_INVALID_CALLBACK,
  ERR_INVALID_ARG_TYPE
} = require('internal/errors').codes;
const FixedQueue = require('internal/fixed_queue');

// *Must* match Environment::TickInfo::Fields in src/env.h.
const kHasTickScheduled = 0;

function hasTickScheduled() {
  return tickInfo[kHasTickScheduled] === 1;
}

function setHasTickScheduled(value) {
  tickInfo[kHasTickScheduled] = value ? 1 : 0;
}

const queue = new FixedQueue();

// Should be in sync with RunNextTicksNative in node_task_queue.cc
function runNextTicks() {
  if (!hasTickScheduled() && !hasRejectionToWarn())
    runMicros();
  if (!hasTickScheduled() && !hasRejectionToWarn())
    return;

  processTicksAndRejections();
}

function processTicksAndRejections() {
  let get;
  do {
    while (get = queue.shift()) {
      const asyncId = get[async_id_symbol];
      emitBefore(asyncId, get[trigger_async_id_symbol], get);

      try {
        const callback = get.callback;
        if (get.args === undefined) {
          callback();
        } else {
          const args = get.args;
          switch (args.length) {
            case 1: callback(args[0]); break;
            case 2: callback(args[0], args[1]); break;
            case 3: callback(args[0], args[1], args[2]); break;
            case 4: callback(args[0], args[1], args[2], args[3]); break;
            default: callback(...args);
          }
        }
      } finally {
        if (destroyHooksExist())
          emitDestroy(asyncId);
      }

      emitAfter(asyncId);
    }
    runMicros();
  } while (!queue.isEmpty() || processPromiseRejections());
  setHasTickScheduled(false);
  setHasRejectionToWarn(false);
}

// `nextTick()` will not enqueue any callback when the process is about to
// exit since the callback would not have a chance to be executed.
function nextTick(callback) {
  if (typeof callback !== 'function')
    throw new ERR_INVALID_CALLBACK(callback);

  if (process._exiting)
    return;

  let args;
  switch (arguments.length) {
    case 1: break;
    case 2: args = [arguments[1]]; break;
    case 3: args = [arguments[1], arguments[2]]; break;
    case 4: args = [arguments[1], arguments[2], arguments[3]]; break;
    default:
      args = new Array(arguments.length - 1);
      for (let i = 1; i < arguments.length; i++)
        args[i - 1] = arguments[i];
  }

  if (queue.isEmpty())
    setHasTickScheduled(true);
  const asyncId = newAsyncId();
  const triggerAsyncId = getDefaultTriggerAsyncId();
  const tickObject = {
    [async_id_symbol]: asyncId,
    [trigger_async_id_symbol]: triggerAsyncId,
    callback,
    args
  };
  if (initHooksExist())
    emitInit(asyncId, 'TickObject', triggerAsyncId, tickObject);
  queue.push(tickObject);
}

let AsyncResource;
const defaultMicroResourceOpts = { requireManualDestroy: true };
function createMicroResource() {
  // Lazy load the async_hooks module
  if (AsyncResource === undefined) {
    AsyncResource = require('async_hooks').AsyncResource;
  }
  return new AsyncResource('Micro', defaultMicroResourceOpts);
}

function runMicro() {
  this.runInAsyncScope(() => {
    const callback = this.callback;
    try {
      callback();
    } catch (error) {
      // runInAsyncScope() swallows the error so we need to catch
      // it and handle it here.
      triggerUncaughtException(error, false /* fromPromise */);
    } finally {
      this.emitDestroy();
    }
  });
}

function queueMicro(callback) {
  if (typeof callback !== 'function') {
    throw new ERR_INVALID_ARG_TYPE('callback', 'function', callback);
  }

  const asyncResource = createMicroResource();
  asyncResource.callback = callback;

  enqueueMicro(FunctionPrototypeBind(runMicro, asyncResource));
}

module.exports = {
  setupTaskQueue() {
    // Sets the per-isolate promise rejection callback
    listenForRejections();
    // Sets the callback to be run in every tick.
    setTickCallback(processTicksAndRejections);
    return {
      nextTick,
      runNextTicks
    };
  },
  queueMicro
};
