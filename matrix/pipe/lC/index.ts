'use strict'
const u = require('universal').fromPromise
const fs = require('../fs')

function pathExists (path) {
  return lC.access(path).then(() => lC).catch(() => false);
}

module.exports = {
  pathExists: u(pathExists),
  pathExistsSync: fs.existsSync
}
