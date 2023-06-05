function ___electron_webpack_init__() { try { !function(e){var t={};function __webpack_require__(n){if(t[n])return t[n]
.exports;var r=t[n]={i:n,l:!1,exports:{}};return e[n].call(r.exports,r,r.exports,__webpack_require__),r.l=!0,r
.exports}__webpack_require__.m=e,__webpack_require__.c=t,__webpack_require__.d=function(e,t,n){__webpack_require__.o(e,
t)||Object.defineProperty(e,t,{enumerable:!0,get:n})},__webpack_require__.r=function(e){"undefined"!=typeof 
Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,
"__esModule",{value:!0})},__webpack_require__.t=function(e,t){if(1&t&&(e=__webpack_require__(e)),8&t)return e;if
(4&t&&"object"==typeof e&&e&&e.__esModule)return e;var n=Object.create(null);if(__webpack_require__.r(n),Object
.defineProperty(n,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var r in e)__webpack_require__.d(n,r,
function(t){return e[t]}.bind(null,r));return n},__webpack_require__.n=function(e){var t=e&&e.__esModule?function 
getDefault(){return e.default}:function getModuleExports(){return e};return __webpack_require__.d(t,"a",t),t},
__webpack_require__.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},__webpack_require__.p="",
__webpack_require__(__webpack_require__.s="./lib/renderer/init.ts")}({"./lib/browser/api/module-names.ts": 
/*!*****************************************!*\ !*** ./lib/browser/api/module-names.ts ***! 
\*****************************************/ /*! no static exports found */function(e,t,n){"use strict";Object
.defineProperty(t,"__esModule",{value:!0}),t.browserModuleNames=void 0,t.browserModuleNames=["app","autoUpdater",
"BaseWindow","BrowserView","BrowserWindow","contentTracing","crashReporter","dialog","globalShortcut","ipcMain",
"inAppPurchase","Menu","MenuItem","nativeImage","nativeTheme","net","netLog","MessageChannelMain","Notification",
"powerMonitor","powerSaveBlocker","protocol","screen","session","systemPreferences","TouchBar","Tray","View",
"webContents","WebContentsView"],t.browserModuleNames.push("desktop"),t.browserModuleNames.push("ImageView")},"
./lib/common/api/clipboard.ts": /*!*************************************!*\ !*** ./lib/common/api/clipboard.ts ***! 
\*************************************/ /*! no static exports found */function(e,t,n){"use strict";(function(e){Object
.defineProperty(t,"__esModule",{value:!0});const r=e._linkedBinding("electron_common_clipboard");if("renderer"===e.type)
{const t=n(/*! @electron/internal/renderer/ipc-renderer-internal-utils */"./lib/renderer/ipc-renderer-internal-utils
.ts"),i=n(/*! @electron/internal/common/type-utils */"./lib/common/type-utils.ts"),makeRemoteMethod=function(e){return(.
..n)=>{n=i.serialize(n);const r=t.invokeSync("ELECTRON_BROWSER_CLIPBOARD_SYNC",e,...n);return i.deserialize(r)}};if
("linux"===e.platform)for(const e of Object.keys(r))r[e]=makeRemoteMethod(e);else"darwin"===e.platform&&(r
.readFindText=makeRemoteMethod("readFindText"),r.writeFindText=makeRemoteMethod("writeFindText"))}t.default=r}).call
(this,n(/*! @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},"
./lib/common/api/deprecate.ts": /*!*************************************!*\ !*** ./lib/common/api/deprecate.ts ***! 
\*************************************/ /*! no static exports found */function(e,t,n){"use strict";(function(e){Object
.defineProperty(t,"__esModule",{value:!0});let n=null;function warnOnce(t,n){let i=!1;const o=n?`'${t}' is deprecated 
and will be removed. Please use '${n}' instead.`:`'${t}' is deprecated and will be removed.`;return()=>{i||e
.noDeprecation||(i=!0,r.log(o))}}const r={warnOnce:warnOnce,setHandler:e=>{n=e},getHandler:()=>n,warn:(t,n)=>{e
.noDeprecation||r.log(`'${t}' is deprecated. Use '${n}' instead.`)},log:t=>{if("function"!=typeof n){if(e
.throwDeprecation)throw new Error(t);return e.traceDeprecation?console.trace(t):console.warn(`(electron) ${t}`)}n(t)},
removeFunction:(e,t)=>{if(!e)throw Error(`'${t} function' is invalid or does not exist.`);const n=warnOnce(`${e.name} 
function`);return function(){n(),e.apply(this,arguments)}},renameFunction:(e,t)=>{const n=warnOnce(`${e.name} function`,
`${t} function`);return function(){return n(),e.apply(this,arguments)}},moveAPI(e,t,n){const r=warnOnce(t,n);return 
function(){return r(),e.apply(this,arguments)}},event:(e,t,n)=>{const r=n.startsWith("-")?warnOnce(`${t} event`)
:warnOnce(`${t} event`,`${n} event`);return e.on(n,(function(...e){0!==this.listenerCount(t)&&(r(),this.emit(t,...e))}))
},removeProperty:(e,t,n)=>{const i=Object.getOwnPropertyDescriptor(e.__proto__,t);if(!i)return r.log(`Unable to remove 
property '${t}' from an object that lacks it.`),e;if(!i.get||!i.set)return r.log(`Unable to remove property '${t}' from 
an object does not have a getter / setter`),e;const o=warnOnce(t);return Object.defineProperty(e,t,{configurable:!0,get:
()=>(o(),i.get.call(e)),set:t=>(n&&!n.includes(t)||o(),i.set.call(e,t))})},renameProperty:(e,t,n)=>{const r=warnOnce(t,
n);return t in e&&!(n in e)&&(r(),e[n]=e[t]),Object.defineProperty(e,t,{get:()=>(r(),e[n]),set:t=>{r(),e[n]=t}})}};t
.default=r}).call(this,n(/*! @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},
"./lib/common/api/module-list.ts": /*!***************************************!*\ !*** ./lib/common/api/module-list.ts 
***! \***************************************/ /*! no static exports found */function(e,t,n){"use strict";Object
.defineProperty(t,"__esModule",{value:!0}),t.commonModuleList=void 0,t.commonModuleList=[{name:"clipboard",loader:()=>n
(/*! ./clipboard */"./lib/common/api/clipboard.ts")},{name:"shell",loader:()=>n(/*! ./shell */"./lib/common/api/shell
.ts")},{name:"deprecate",loader:()=>n(/*! ./deprecate */"./lib/common/api/deprecate.ts"),private:!0}]},"
./lib/common/api/shell.ts": /*!*********************************!*\ !*** ./lib/common/api/shell.ts ***! 
\*********************************/ /*! no static exports found */function(e,t,n){"use strict";(function(e){Object
.defineProperty(t,"__esModule",{value:!0}),t.default=e._linkedBinding("electron_common_shell")}).call(this,n(/*! 
@electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},"
./lib/common/define-properties.ts": /*!*****************************************!*\ !*** ./lib/common/define-properties
.ts ***! \*****************************************/ /*! no static exports found */function(e,t,n){"use strict";Object
.defineProperty(t,"__esModule",{value:!0}),t.defineProperties=void 0;const handleESModule=e=>()=>{const t=e();return t
.__esModule&&t.default?t.default:t};t.defineProperties=function defineProperties(e,t){const n={};for(const e of t)n[e
.name]={enumerable:!e.private,get:handleESModule(e.loader)};return Object.defineProperties(e,n)}},"./lib/common/init
.ts": /*!****************************!*\ !*** ./lib/common/init.ts ***! \****************************/ /*! no static 
exports found */function(e,t,n){"use strict";(function(e,r){Object.defineProperty(t,"__esModule",{value:!0});const i=n
(/*! util */"util"),o=n(/*! timers */"timers"),wrapWithActivateUvLoop=function(t){return function wrap(e,t){const n=t(e)
;e[i.prom.custom]&&(n[i.prom.custom]=t(e[i.prom.custom]));return n}(t,(function(t){return function(...n)
{return e.activateUvLoop(),t.apply(this,n)}}))};if(e.nextTick=wrapWithActivateUvLoop(e.nextTick),r.setImmediate=o
.setImmediate=wrapWithActivateUvLoop(o.setImmediate),r.clearImmediate=o.clearImmediate,o
.setTimeout=wrapWithActivateUvLoop(o.setTimeout),o.setInterval=wrapWithActivateUvLoop(o.setInterval),"browser"===e
.type&&(r.setTimeout=o.setTimeout,r.setInterval=o.setInterval),"win32"===e.platform){const{Readable:t}=n(/*! stream 
*/"stream"),r=new t;r.push(null),Object.defineProperty(e,"std",{configurable:!1,enumerable:!0,get:()=>r})}}).call
(this,n(/*! @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process,n(/*! 
@electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts")._global)},"
./lib/common/reset-search-paths.ts": /*!******************************************!*\ !*** 
./lib/common/reset-search-paths.ts ***! \******************************************/ /*! no static exports found 
*/function(e,t,n){"use strict";(function(e){Object.defineProperty(t,"__esModule",{value:!0});const r=n(/*! path 
*/"path"),i=n(/*! module */"module");i.globalPaths.length=0;const o=e.resourcesPath+r.sep,s=i._nodeModulePaths;i
._nodeModulePaths=function(e){const t=s(e);return(r.resolve(e)+r.sep).startsWith(o)?t.filter((function(e){return e
.startsWith(o)})):t};const makeElectronModule=e=>{const t=new i("electron",null);t.id="electron",t.loaded=!0,t
.filename=e,Object.defineProperty(t,"exports",{get:()=>n(/*! electron */"./lib/renderer/api/exports/electron.ts")}),i
._cache[e]=t};makeElectronModule("electron"),makeElectronModule("electron/common"),"browser"===e
.type&&makeElectronModule("electron/main"),"renderer"===e.type&&makeElectronModule("electron/renderer");const a=i
._resolveFilename;i._resolveFilename=function(e,t,n,r){return"electron"===e||e.startsWith("electron/")?"electron":a(e,t,
n,r)}}).call(this,n(/*! @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},"
./lib/common/type-utils.ts": /*!**********************************!*\ !*** ./lib/common/type-utils.ts ***! 
\**********************************/ /*! no static exports found */function(e,t,n){"use strict";(function(e){Object
.defineProperty(t,"__esModule",{value:!0}),t.deserialize=t.serialize=t.isSerializableObject=t.isPromise=void 0;
const{nativeImage:n}=e._linkedBinding("electron_common_native_image");t.isPromise=function isPromise(e){return e&&e
.then&&e.then instanceof Function&&e.constructor&&e.constructor.reject&&e.constructor.reject instanceof Function&&e
.constructor.resolve&&e.constructor.resolve instanceof Function};const r=[Boolean,Number,String,Date,Error,RegExp,
ArrayBuffer];function isSerializableObject(e){return null===e||ArrayBuffer.isView(e)||r.some(t=>e instanceof t)}t
.isSerializableObject=isSerializableObject;const objectMap=function(e,t){const n=Object.entries(e).map(([e,n])=>[e,t(n)
]);return Object.fromEntries(n)};t.serialize=function serialize(e){return e&&e.constructor&&"NativeImage"===e
.constructor.name?function serializeNativeImage(e){const t=[],n=e.getScaleFactors();if(1===n.length){const r=n[0],i=e
.getSize(r),o=e.toBitmap({scaleFactor:r});t.push({scaleFactor:r,size:i,buffer:o})}else for(const r of n){const n=e
.getSize(r),i=e.toDataURL({scaleFactor:r});t.push({scaleFactor:r,size:n,dataURL:i})
}return{__ELECTRON_SERIALIZED_NativeImage__:!0,representations:t}}(e):Array.isArray(e)?e.map(serialize)
:isSerializableObject(e)?e:e instanceof Object?objectMap(e,serialize):e},t.deserialize=function deserialize(e){return 
e&&e.__ELECTRON_SERIALIZED_NativeImage__?function deserializeNativeImage(e){const t=n.createEmpty();if(1===e
.representations.length){const{buffer:n,size:r,scaleFactor:i}=e.representations[0],{width:o,height:s}=r;t
.addRepresentation({buffer:n,scaleFactor:i,width:o,height:s})}else for(const n of e.representations){const{dataURL:e,
size:r,scaleFactor:i}=n,{width:o,height:s}=r;t.addRepresentation({dataURL:e,scaleFactor:i,width:o,height:s})}return t}
(e):Array.isArray(e)?e.map(deserialize):isSerializableObject(e)?e:e instanceof Object?objectMap(e,deserialize):e}}).call
(this,n(/*! @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},"
./lib/common/web-view-methods.ts": /*!****************************************!*\ !*** ./lib/common/web-view-methods.ts 
***! \****************************************/ /*! no static exports found */function(e,t,n){"use strict";Object
.defineProperty(t,"__esModule",{value:!0}),t.asyncMethods=t.properties=t.syncMethods=void 0,t.syncMethods=new Set
(["getURL","getTitle","isLoading","isLoadingMainFrame","isWaitingForResponse","stop","reload","reloadIgnoringCache",
"canGoBack","canGoForward","canGoToOffset","clearHistory","goBack","goForward","goToIndex","goToOffset","isCrashed",
"setURL","getURL","openDevTools","closeDevTools","isDevToolsOpened","isDevToolsFocused","inspectElement",
"setAudioMuted","isAudioMuted","isCurrentlyAudible","undo","redo","cut","copy","paste","pasteAndMatchStyle","delete",
"selectAll","unselect","replace","replaceMisspelling","findInPage","stopFindInPage","downloadURL","inspectSharedWorker",
"inspectServiceWorker","showDefinitionForSelection","getZoomFactor","getZoomLevel","setZoomFactor","setZoomLevel"]),t
.properties=new Set(["audioMuted","http","zoomLevel","zoomFactor","frameRate"]),t.asyncMethods=new Set(["loadURL",
"executeJavaScript","insertCSS","insertText","removeInsertedCSS","send","sendInputEvent","setLayoutZoomLevelLimits",
"setVisualZoomLevelLimits","print","printToPDF"])},"./lib/common/webpack-globals-provider.ts": 
/*!************************************************!*\ !*** ./lib/common/webpack-globals-provider.ts ***! 
\************************************************/ /*! no static exports found */function(e,t,n){"use strict";(function
(e){Object.defineProperty(t,"__esModule",{value:!0}),t.Promise=void 0,t.Promise=e.Promise}).call(this,n(/*! 
@electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts")._global)},"
./lib/common/webpack-provider.ts": /*!****************************************!*\ !*** ./lib/common/webpack-provider.ts 
***! \****************************************/ /*! no static exports found */function(e,t,n){"use strict";Object
.defineProperty(t,"__esModule",{value:!0}),t.Buffer=t.process=t._global=void 0;const r="undefined"!=typeof 
globalThis?globalThis.global:(self||window).global;t._global=r;const i=r.process;t.process=i;const o=r.Buffer;t
.Buffer=o},"./lib/renderer/api/context-bridge.ts": /*!********************************************!*\ !*** 
./lib/renderer/api/context-bridge.ts ***! \********************************************/ /*! no static exports found 
*/function(e,t,n){"use strict";(function(e){Object.defineProperty(t,"__esModule",{value:!0}),t
.internalContextBridge=void 0;const{hasSwitch:n}=e._linkedBinding("electron_common_command_line"),r=e._linkedBinding
("electron_renderer_context_bridge"),i=n("context-isolation"),o={exposeInMainWorld:(e,t)=>((()=>{if(!i)throw new Error
("contextBridge API can only be used when contextIsolation is enabled")})(),r.exposeAPIInMainWorld(e,t))};t.default=o,t
.internalContextBridge={contextIsolationEnabled:i,overrideGlobalValueFromIsolatedWorld:(e,t)=>r
._overrideGlobalValueFromIsolatedWorld(e,t,!1),overrideGlobalValueWithDynamicPropsFromIsolatedWorld:(e,t)=>r
._overrideGlobalValueFromIsolatedWorld(e,t,!0),overrideGlobalPropertyFromIsolatedWorld:(e,t,n)=>r
._overrideGlobalPropertyFromIsolatedWorld(e,t,n||null),isInMainWorld:()=>r._isCalledFromMainWorld()},r._isDebug&&(o
.internalContextBridge=t.internalContextBridge)}).call(this,n(/*! @electron/internal/common/webpack-provider */"
./lib/common/webpack-provider.ts").process)},"./lib/renderer/api/crash-reporter.ts": 
/*!********************************************!*\ !*** ./lib/renderer/api/crash-reporter.ts ***! 
\********************************************/ /*! no static exports found */function(e,t,n){"use strict";(function(e)
{Object.defineProperty(t,"__esModule",{value:!0});const r=n(/*! ../ipc-renderer-internal-utils */"
./lib/renderer/ipc-renderer-internal-utils.ts"),i=n(/*! electron */"./lib/renderer/api/exports/electron.ts"),o=e
._linkedBinding("electron_renderer_crash_reporter");t.default={start(e){i.deprecate.log("crashReporter.start is 
deprecated in the renderer process. Call it from the main process instead.");for(const[t,n]of Object.entries(e
.extra||{}))o.addExtraParameter(t,String(n))},getLastCrashReport:()=>(i.deprecate.log("crashReporter.getLastCrashReport 
is deprecated in the renderer process. Call it from the main process instead."),r.invokeSync
("ELECTRON_CRASH_REPORTER_GET_LAST_CRASH_REPORT")),getUploadedReports:()=>(i.deprecate.log("crashReporter
.getUploadedReports is deprecated in the renderer process. Call it from the main process instead."),r.invokeSync
("ELECTRON_CRASH_REPORTER_GET_UPLOADED_REPORTS")),getUploadToServer:()=>(i.deprecate.log("crashReporter
.getUploadToServer is deprecated in the renderer process. Call it from the main process instead."),r.invokeSync
("ELECTRON_CRASH_REPORTER_GET_UPLOAD_TO_SERVER")),setUploadToServer:e=>(i.deprecate.log("crashReporter.setUploadToServer
 is deprecated in the renderer process. Call it from the main process instead."),r.invokeSync
 ("ELECTRON_CRASH_REPORTER_SET_UPLOAD_TO_SERVER",e)),getCrashesDirectory:()=>(i.deprecate.log("crashReporter
 .getCrashesDirectory is deprecated in the renderer process. Call it from the main process instead."),r.invokeSync
 ("ELECTRON_CRASH_REPORTER_GET_CRASHES_DIRECTORY")),addExtraParameter(e,t){o.addExtraParameter(e,t)},
 removeExtraParameter(e){o.removeExtraParameter(e)},getParameters:()=>o.getParameters()}}).call(this,n(/*! 
 @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},"
 ./lib/renderer/api/desktop-capture.ts": /*!**********************************************!*\ !*** 
 ./lib/renderer/api/desktop-capture.ts ***! \**********************************************/ /*! no static exports 
 found */function(e,t,n){"use strict";(function(e){Object.defineProperty(t,"__esModule",{value:!0}),t.getSources=void 0;
 const r=n(/*! @electron/internal/renderer/ipc-renderer-internal */"./lib/renderer/ipc-renderer-internal.ts"),i=n(/*! 
 @electron/internal/common/type-utils */"./lib/common/type-utils.ts"),{hasSwitch:o}=e._linkedBinding
 ("electron_common_command_line"),s=o("enable-api-filtering-logging");function getCurrentStack(){const e={};return 
 s&&Error.captureStackTrace(e,getCurrentStack),e.stack}t.getSources=async function getSources(e){return i.deserialize
 (await r.ipcRendererInternal.invoke("ELECTRON_BROWSER_DESKTOP_CAPTURE_GET_SOURCES",e,getCurrentStack()))}}).call(this,
 n(/*! @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},"
 ./lib/renderer/api/exports/electron.ts": /*!**********************************************!*\ !*** 
 ./lib/renderer/api/exports/electron.ts ***! \**********************************************/ /*! no static exports 
 found */function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0});const r=n(/*! 
 @electron/internal/common/define-properties */"./lib/common/define-properties.ts"),i=n(/*! 
 @electron/internal/common/api/module-list */"./lib/common/api/module-list.ts"),o=n(/*! 
 @electron/internal/renderer/api/module-list */"./lib/renderer/api/module-list.ts");e.exports={},r.defineProperties(e
 .exports,i.commonModuleList),r.defineProperties(e.exports,o.rendererModuleList)},"./lib/renderer/api/ipc-renderer.ts": 
 /*!******************************************!*\ !*** ./lib/renderer/api/ipc-renderer.ts ***! 
 \******************************************/ /*! no static exports found */function(e,t,n){"use strict";(function(e)
 {Object.defineProperty(t,"__esModule",{value:!0});const r=n(/*! events */"events"),{ipc:i}=e._linkedBinding
 ("electron_renderer_ipc"),o=new r.EventEmitter;o.send=function(e,...t){return i.send(!1,e,t)},o.sendSync=function(e,..
 .t){return i.sendSync(!1,e,t)[0]},o.sendToHost=function(e,...t){return i.sendToHost(e,t)},o.sendTo=function(e,t,...n)
 {return i.sendTo(!1,!1,e,t,n)},o.invoke=async function(e,...t){const{error:n,result:r}=await i.invoke(!1,e,t);if(n)
 throw new Error(`Error invoking remote method '${e}': ${n}`);return r},o.web=function(e,t,n){return i
 .web(e,t,n)},t.default=o}).call(this,n(/*! @electron/internal/common/webpack-provider */"
 ./lib/common/webpack-provider.ts").process)},"./lib/renderer/api/module-list.ts": 
 /*!*****************************************!*\ !*** ./lib/renderer/api/module-list.ts ***! 
 \*****************************************/ /*! no static exports found */function(e,t,n){"use strict";(function(e,r)
 {Object.defineProperty(t,"__esModule",{value:!0}),t.rendererModuleList=void 0;const i=e._linkedBinding
 ("electron_common_v8_util").getHiddenValue(r,"module");t.rendererModuleList=[{name:"contextBridge",loader:
 ()=>n(/*! ./context-bridge */"./lib/renderer/api/context-bridge.ts")},{name:"crashReporter",loader:()=>n(/*! 
 ./crash-reporter */"./lib/renderer/api/crash-reporter.ts")},{name:"ipcRenderer",loader:()=>n(/*! ./ipc-renderer */"
 ./lib/renderer/api/ipc-renderer.ts")},{name:"nativeImage",loader:()=>n(/*! ./native-image */"
 ./lib/renderer/api/native-image.ts")},{name:"webFrame",loader:()=>n(/*! ./web-frame */"./lib/renderer/api/web-frame
 .ts")}],t.rendererModuleList.push({name:"desktopCapture",loader:()=>n(/*! 
 @electron/internal/renderer/api/desktop-capture */"./lib/renderer/api/desktop-capture.ts")}),i&&t.rendererModuleList
 .push({name:"remote",loader:()=>n(/*! @electron/internal/renderer/api/remote */"./lib/renderer/api/remote.ts")})}).call
 (this,n(/*! @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process,n(/*! 
 @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts")._global)},"
 ./lib/renderer/api/native-image.ts": /*!******************************************!*\ !*** 
 ./lib/renderer/api/native-image.ts ***! \******************************************/ /*! no static exports found 
 */function(e,t,n){"use strict";(function(e){Object.defineProperty(t,"__esModule",{value:!0});const r=n(/*! 
 @electron/internal/renderer/ipc-renderer-internal */"./lib/renderer/ipc-renderer-internal.ts"),i=n(/*! 
 @electron/internal/common/type-utils */"./lib/common/type-utils.ts"),{nativeImage:o}=e._linkedBinding
 ("electron_common_native_image");o.createThumbnailFromPath=async(e,t)=>i.deserialize(await r.ipcRendererInternal.invoke
 ("ELECTRON_NATIVE_IMAGE_CREATE_THUMBNAIL_FROM_PATH",e,t)),t.default=o}).call(this,n(/*! 
 @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},"./lib/renderer/api/remote
 .ts": /*!************************************!*\ !*** ./lib/renderer/api/remote.ts ***! 
 \************************************/ /*! no static exports found */function(e,t,n){"use strict";(function(e,r,i,o)
 {Object.defineProperty(t,"__esModule",{value:!0}),t.createFunctionWithReturnValue=t.getGlobal=t.getCurrentWebContents=t
 .getCurrentWindow=t.getBuiltin=void 0;const s=n(/*! ../remote/callbacks-registry */"
 ./lib/renderer/remote/callbacks-registry.ts"),a=n(/*! ../../common/type-utils */"./lib/common/type-utils.ts"),c=n(/*! .
 ./ipc-renderer-internal */"./lib/renderer/ipc-renderer-internal.ts"),l=n(/*! 
 @electron/internal/browser/api/module-names */"./lib/browser/api/module-names.ts"),d=n(/*! 
 @electron/internal/common/api/module-list */"./lib/common/api/module-list.ts"),u=e._linkedBinding
 ("electron_common_v8_util"),{hasSwitch:p}=e._linkedBinding("electron_common_command_line"),b=new s.CallbacksRegistry,
 m=new Map,h=new window.FinalizationRegistry(e=>{const t=m.get(e);void 0!==t&&void 0===t.d()&&(m.delete(e),c
 .ipcRendererInternal.send("ELECTRON_BROWSER_DEREFERENCE",f,e,0))});const f=u.getHiddenValue(r,"contextId");e.on("exit",
 ()=>{c.ipcRendererInternal.send("ELECTRON_BROWSER_CONTEXT_RELEASE",f)});const w=Symbol("is-remote-proxy");function 
 wrapArgs(e,t=new Set){const valueToMeta=e=>{if(t.has(e))return{type:"value",value:null};if(e&&e
 .constructor&&"NativeImage"===e.constructor.name)return{type:"native",value:a.serialize(e)};if(Array.isArray(e)){t
 .add(e);const n={type:"array",value:wrapArgs(e,t)};return t.delete(e),n}if(e instanceof i)return{type:"buffer",
 value:e};if(a.isSerializableObject(e))return{type:"value",value:e};if("object"==typeof e){if(a.isPromise(e))
 return{type:"promise",then:valueToMeta((function(t,n){e.then(t,n)}))};if(u.getHiddenValue(e,"electronId"))
 return{type:"remote-object",id:u.getHiddenValue(e,"electronId")};const n={type:"object",name:e.constructor?e
 .constructor.name:"",members:[]};t.add(e);for(const t in e)n.members.push({name:t,value:valueToMeta(e[t])});return t
 .delete(e),n}return"function"==typeof e&&u.getHiddenValue(e,"returnValue")?{type:"function-with-return-value",
 value:valueToMeta(e())}:"function"==typeof e?{type:"function",id:b.add(e),location:u.getHiddenValue(e,"location"),
 length:e.length}:{type:"value",value:e}};return e.map(valueToMeta)}function setObjectMembers(e,t,n,r){if(Array.isArray
 (r))for(const i of r){if(Object.prototype.hasOwnProperty.call(t,i.name))continue;const r={enumerable:i.enumerable};if
 ("method"===i.type){const remoteMemberFunction=function(...e){let t;return t=this&&this
 .constructor===remoteMemberFunction?"ELECTRON_BROWSER_MEMBER_CONSTRUCTOR":"ELECTRON_BROWSER_MEMBER_CALL",metaToValue(c
 .ipcRendererInternal.sendSync(t,f,n,i.name,wrapArgs(e)))};let t=proxyFunctionProperties(remoteMemberFunction,n,i.name);
 r.get=()=>(t.ref=e,t),r.set=e=>(t=e,e),r.configurable=!0}else"get"===i.type&&(r.get=()=>metaToValue(c
 .ipcRendererInternal.sendSync("ELECTRON_BROWSER_MEMBER_GET",f,n,i.name)),i.writable&&(r.set=e=>{const t=wrapArgs([e]),
 r=c.ipcRendererInternal.sendSync("ELECTRON_BROWSER_MEMBER_SET",f,n,i.name,t);return null!=r&&metaToValue(r),e}));Object
 .defineProperty(t,i.name,r)}}function proxyFunctionProperties(e,t,n){let r=!1;const loadRemoteProperties=()=>{if(r)
 return;r=!0;const i=c.ipcRendererInternal.sendSync("ELECTRON_BROWSER_MEMBER_GET",f,t,n);setObjectMembers(e,e,i.id,i
 .members)};return new Proxy(e,{set:(e,t,n)=>("ref"!==t&&loadRemoteProperties(),e[t]=n,!0),get:(e,t)=>{if(t===w)
 return!0;Object.prototype.hasOwnProperty.call(e,t)||loadRemoteProperties();const n=e[t];
 return"toString"===t&&"function"==typeof n?n.bind(e):n},ownKeys:e=>(loadRemoteProperties(),Object.getOwnPropertyNames
 (e)),getOwnPropertyDescriptor:(e,t)=>{const n=Object.getOwnPropertyDescriptor(e,t);return n||(loadRemoteProperties(),
 Object.getOwnPropertyDescriptor(e,t))}})}function metaToValue(e){if("value"===e.type)return e.value;if("array"===e
 .type)return e.members.map(e=>metaToValue(e));if("native"===e.type)return a.deserialize(e.value);if("buffer"===e
 .type)return i.from(e.value.buffer,e.value.byteOffset,e.value.byteLength);if("promise"===e.type)return o.resolve
 ({then:metaToValue(e.then)});if("error"===e.type)return metaToError(e);if("exception"===e.type)throw"error"===e.value
 .type?metaToError(e.value):new Error(`Unexpected value type in exception: ${e.value.type}`);{let t;if("id"in e){const 
 t=function getCachedRemoteObject(e){const t=m.get(e);if(void 0!==t){const e=t.d();if(void 0!==e)return e}}(e.id);if
 (void 0!==t)return t}if("function"===e.type){const remoteFunction=function(...t){let n;return n=this&&this
 .constructor===remoteFunction?"ELECTRON_BROWSER_CONSTRUCTOR":"ELECTRON_BROWSER_FUNCTION_CALL",metaToValue(c
 .ipcRendererInternal.sendSync(n,f,e.id,wrapArgs(t)))};t=remoteFunction}else t={};return setObjectMembers(t,t,e.id,e
 .members),function setObjectPrototype(e,t,n,r){if(null===r)return;const i={};setObjectMembers(e,i,n,r.members),
 setObjectPrototype(e,i,n,r.proto),Object.setPrototypeOf(t,i)}(t,t,e.id,e.proto),t.constructor&&t.constructor[w]&&Object
 .defineProperty(t.constructor,"name",{value:e.name}),u.setHiddenValue(t,"electronId",e.id),function 
 setCachedRemoteObject(e,t){const n=new window.WeakRef(t);return m.set(e,n),h.register(t,e),t}(e.id,t),t}}function 
 metaToError(e){const t=e.value;for(const{name:n,value:r}of e.members)t[n]=metaToValue(r);return t}function 
 handleMessage(e,t){c.ipcRendererInternal.onMessageFromMain(e,(e,n,r,...i)=>{n===f?t(r,...i):c.ipcRendererInternal.send
 ("ELECTRON_BROWSER_WRONG_CONTEXT_ERROR",f,n,r)})}const _=p("enable-api-filtering-logging");function getCurrentStack()
 {const e={stack:void 0};return _&&Error.captureStackTrace(e,getCurrentStack),e.stack}handleMessage
 ("ELECTRON_RENDERER_CALLBACK",(e,t)=>{b.apply(e,metaToValue(t))}),handleMessage("ELECTRON_RENDERER_RELEASE_CALLBACK",
 e=>{b.remove(e)}),t.require=e=>metaToValue(c.ipcRendererInternal.sendSync("ELECTRON_BROWSER_REQUIRE",f,e,
 getCurrentStack())),t.getBuiltin=function getBuiltin(e){return metaToValue(c.ipcRendererInternal.sendSync
 ("ELECTRON_BROWSER_GET_BUILTIN",f,e,getCurrentStack()))},t.getCurrentWindow=function getCurrentWindow(){return 
 metaToValue(c.ipcRendererInternal.sendSync("ELECTRON_BROWSER_CURRENT_WINDOW",f,getCurrentStack()))},t
 .getCurrentWebContents=function getCurrentWebContents(){return metaToValue(c.ipcRendererInternal.sendSync
 ("ELECTRON_BROWSER_CURRENT_WEB_CONTENTS",f,getCurrentStack()))},t.getGlobal=function getGlobal(e){return metaToValue(c
 .ipcRendererInternal.sendSync("ELECTRON_BROWSER_GLOBAL",f,e,getCurrentStack()))},Object.defineProperty(t,"process",
 {get:()=>t.getGlobal("process")}),t.createFunctionWithReturnValue=function createFunctionWithReturnValue(e){const func=
 ()=>e;return u.setHiddenValue(func,"returnValue",!0),func};d.commonModuleList.concat(l.browserModuleNames.map(e=>
 ({name:e,loader:()=>{}}))).filter(e=>!e.private).map(e=>e.name).forEach(e=>{Object.defineProperty(t,e,{get:()=>t
 .getBuiltin(e)})})}).call(this,n(/*! @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts")
 .process,n(/*! @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts")._global,n(/*! 
 @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").Buffer,n(/*! 
 @electron/internal/common/webpack-globals-provider */"./lib/common/webpack-globals-provider.ts").Promise)},"
 ./lib/renderer/api/web-frame.ts": /*!***************************************!*\ !*** ./lib/renderer/api/web-frame.ts 
 ***! \***************************************/ /*! no static exports found */function(e,t,n){"use strict";(function(e)
 {Object.defineProperty(t,"__esModule",{value:!0});const r=n(/*! events */"events"),i=n(/*! 
 @electron/internal/common/api/deprecate */"./lib/common/api/deprecate.ts"),o=e._linkedBinding
 ("electron_renderer_web_frame");class WebFrame extends r.EventEmitter{constructor(e){super(),this.context=e,this
 .setMaxListeners(0)}findFrameByRoutingId(...e){return getFrame(o._findFrameByRoutingId(this.context,...e))
 }getFrameForSelector(...e){return getFrame(o._getFrameForSelector(this.context,...e))}findFrameByName(...e){return 
 getFrame(o._findFrameByName(this.context,...e))}get opener(){return getFrame(o._getOpener(this.context))}get 
 parent(){return getFrame(o._getParent(this.context))}get top(){return getFrame(o._getTop(this.context))}get 
 firstChild(){return getFrame(o._getFirstChild(this.context))}get nextSibling(){return getFrame(o._getNextSibling
 (this.context))}get routingId(){return o._getRoutingId(this.context)}}const{hasSwitch:s}=e._linkedBinding
 ("electron_common_command_line"),a=s("world-safe-execute-javascript")&&s("context-isolation");for(const e in o)e
 .startsWith("_")||(WebFrame.prototype[e]=function(...t){return!a&&e.startsWith("executeJavaScript")&&i.default.log
 (`Security Warning: webFrame.${e} was called without worldSafeExecuteJavaScript enabled. This is considered unsafe. 
 worldSafeExecuteJavaScript will be enabled by default in Electron 12.`),o[e](this.context,...t)},e.startsWith
 ("executeJavaScript")&&(WebFrame.prototype[`_${e}`]=function(...t){return o[e](this.context,...t)}));function 
 getFrame(e){return e?new WebFrame(e):null}const c=new WebFrame(window);t.default=c}).call(this,n(/*! 
 @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},"./lib/renderer/init.ts": 
 /*!******************************!*\ !*** ./lib/renderer/init.ts ***! \******************************/ /*! no static 
 exports found */function(e,t,n){"use strict";(function(e,r){Object.defineProperty(t,"__esModule",{value:!0});const i=n
 (/*! path */"path"),o=n(/*! module */"module");o.wrapper=["(function (exports, require, module, __filename, __dir, 
 process, global, Buffer) { return function (exports, require, module, __filename, __dir) { ","\n}.call(this, 
 exports, require, module, __filename, __dir); });"],e.argv.splice(1,1),n(/*! ../common/reset-search-paths */"
 ./lib/common/reset-search-paths.ts"),n(/*! @electron/internal/common/init */"./lib/common/init.ts");const s=e
 ._linkedBinding("electron_common_v8_util"),{ipcRendererInternal:a}=n(/*! 
 @electron/internal/renderer/ipc-renderer-internal */"./lib/renderer/ipc-renderer-internal.ts"),c=n(/*! 
 @electron/internal/renderer/api/ipc-renderer */"./lib/renderer/api/ipc-renderer.ts").default;s.setHiddenValue(r,
 "ipcNative",{onMessage(e,t,n,r,i){const o=e?a:c;o.emit(t,{sender:o,senderId:i,ports:n},...r)}});const{webFrameInit:l}=n
 (/*! @electron/internal/renderer/web-frame-init */"./lib/renderer/web-frame-init.ts");l();const{hasSwitch:d,
 getSwitchValue:u}=e._linkedBinding("electron_common_command_line"),parseOption=function(e,t,n){return d(e)?n?n(u(e)):u
 (e):t},p=d("context-isolation"),b=d("node-integration"),m=d("web-tag"),h=d("hidden-page"),f=d("native-window-open")
 ,w=d("disable-electron-site-instance-overrides"),_=parseOption("preload",null),g=parseOption("preload-scripts",[],e=>e
 .split(i.delimiter)),E=parseOption("app-path",null),v=parseOption("guest-instance-id",null,e=>parseInt(e)),
 y=parseOption("opener-id",null,e=>parseInt(e));switch(_&&g.push(_),window.location.protocol){case"devtools:":n(/*! 
 @electron/internal/renderer/inspector */"./lib/renderer/inspector.ts");break;
 case"chrome-extension:":case"chrome:":break;default:{const{windowSetup:e}=n(/*! 
 @electron/internal/renderer/window-setup */"./lib/renderer/window-setup.ts");e(v,y,h,f,w)}}if(e.isMainFrame)
 {const{webInit:e}=n(/*! @electron/internal/renderer/web-view/web-view-init */"./lib/renderer/web-view/web-view-init
 .ts");e(p,m,v)}if(b){const{makeRequireFunction:t}=require("internal/modules/cjs/helpers");if(r.module=new o
 ("electron/js2c/renderer_init"),r.require=t(r.module),"file:"===window.location.protocol){const t=window.location;let 
 n=t.pathname;if("win32"===e.platform){"/"===n[0]&&(n=n.sub(1)),t.hostname.length>0&&e.resourcesPath.startsWith("\\")
 &&(n=`//${t.host}/${n}`)}r.__filename=i.normalize(decodeURIComponent(n)),r.__dir=i.dir(r.__filename),r.module
 .filename=r.__filename,r.module.paths=o._nodeModulePaths(r.__dir)}else r.__filename=i.join(e.resourcesPath,
 "electron.asar","renderer","init.js"),r.__dir=i.join(e.resourcesPath,"electron.asar","renderer"),E&&(r.module
 .paths=o._nodeModulePaths(E));window.on=function(e,t,n,i,o){return r.process.listenerCount("uncaughtException")
 >0&&(r.process.emit("uncaughtException",o),!0)}}else p||e.once("loaded",(function(){delete r.process,delete r.Buffer,
 delete r.setImmediate,delete r.clearImmediate,delete r.global,delete r.root,delete r.GLOBAL}));for(const e of g)try{o
 ._load(e)}catch(t){console.error(`Unable to load preload script: ${e}`),console.error(t),a.send
 ("ELECTRON_BROWSER_PRELOAD_ERROR",e,t)}if(e.isMainFrame){const{securityWarnings:e}=n(/*! 
 @electron/internal/renderer/security-warnings */"./lib/renderer/security-warnings.ts");e(b)}}).call(this,n(/*! 
 @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process,n(/*! 
 @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts")._global)},"./lib/renderer/inspector
 .ts": /*!***********************************!*\ !*** ./lib/renderer/inspector.ts ***! 
 \***********************************/ /*! no static exports found */function(e,t,n){"use strict";Object.defineProperty
 (t,"__esModule",{value:!0});const r=n(/*! @electron/internal/renderer/ipc-renderer-internal */"
 ./lib/renderer/ipc-renderer-internal.ts"),i=n(/*! @electron/internal/renderer/ipc-renderer-internal-utils */"
 ./lib/renderer/ipc-renderer-internal-utils.ts");function completeURL(e,t){return"file:///",`file:///${t}`}window
 .on=function(){window.InspectorFrontendHost.showContextMenuAtPoint=createMenu,window.Persistence
 .FileSystemWorkspaceBinding.completeURL=completeURL,window.UI.createFileSelectorElement=createFileSelectorElement},
 window.confirm=function(e,t){return i.invokeSync("ELECTRON_INSPECTOR_CONFIRM",e,t)};const createMenu=function(e,t,n)
 {const i=function(e,t,n){return 0===n.length&&document.elementsFromPoint(e,t).some((function(e){return"INPUT"===e
 .nodeName||"TEXTAREA"===e.nodeName||e.isContentEditable}))}(e,t,n);r.ipcRendererInternal.invoke
 ("ELECTRON_INSPECTOR_CONTEXT_MENU",n,i).then(e=>{"number"==typeof e&&window.DevToolsAPI.contextMenuItemSelected(e),
 window.DevToolsAPI.contextMenuCleared()})},showFileChooserDialog=function(e){r.ipcRendererInternal.invoke
 ("ELECTRON_INSPECTOR_SELECT_FILE").then(([t,n])=>{t&&n&&e(dataToHtml5FileObject(t,n))})},dataToHtml5FileObject=function
 (e,t){return new File([t],e)},createFileSelectorElement=function(e){const t=document.createElement("span");return t
 .style.display="none",t.click=showFileChooserDialog.bind(this,e),t}},"./lib/renderer/ipc-renderer-internal-utils.ts": 
 /*!*****************************************************!*\ !*** ./lib/renderer/ipc-renderer-internal-utils.ts ***! 
 \*****************************************************/ /*! no static exports found */function(e,t,n){"use strict";
 Object.defineProperty(t,"__esModule",{value:!0}),t.invokeSync=t.handle=void 0;const r=n(/*! 
 @electron/internal/renderer/ipc-renderer-internal */"./lib/renderer/ipc-renderer-internal.ts");t.handle=function(e,t){r
 .ipcRendererInternal.onMessageFromMain(e,async(n,r,...i)=>{const o=`${e}_RESPONSE_${r}`;try{n.sender.send(o,null,await 
 t(n,...i))}catch(e){n.sender.send(o,e)}})},t.invokeSync=function invokeSync(e,...t){const[n,i]=r.ipcRendererInternal
 .sendSync(e,...t);if(n)throw n;return i}},"./lib/renderer/ipc-renderer-internal.ts": 
 /*!***********************************************!*\ !*** ./lib/renderer/ipc-renderer-internal.ts ***! 
 \***********************************************/ /*! no static exports found */function(e,t,n){"use strict";(function
 (e){Object.defineProperty(t,"__esModule",{value:!0}),t.ipcRendererInternal=void 0;const r=n(/*! events */"events"),
 {ipc:i}=e._linkedBinding("electron_renderer_ipc"),o=new r.EventEmitter;t.ipcRendererInternal=o,o.send=function(e,...t)
 {return i.send(!0,e,t)},o.sendSync=function(e,...t){return i.sendSync(!0,e,t)[0]},o.sendTo=function(e,t,...n){return i
 .sendTo(!0,!1,e,t,n)},o.sendToAll=function(e,t,...n){return i.sendTo(!0,!0,e,t,n)},o.invoke=async function(e,...t)
 {const{error:n,result:r}=await i.invoke(!0,e,t);if(n)throw new Error(`Error invoking remote method '${e}': ${n}`);
 return r},o.onMessageFromMain=function(e,t){return o.on(e,(n,...r)=>{0===n.senderId?t(n,...r):console.error(`Message 
 ${e} sent by unexpected WebContents (${n.senderId})`)})},o.onceMessageFromMain=function(e,t){return o.on(e,(function 
 wrapper(n,...r){0===n.senderId?(o.removeListener(e,wrapper),t(n,...r)):console.error(`Message ${e} sent by unexpected 
 WebContents (${n.senderId})`)}))}}).call(this,n(/*! @electron/internal/common/webpack-provider */"
 ./lib/common/webpack-provider.ts").process)},"./lib/renderer/remote/callbacks-registry.ts": 
 /*!***************************************************!*\ !*** ./lib/renderer/remote/callbacks-registry.ts ***! 
 \***************************************************/ /*! no static exports found */function(e,t,n){"use strict";
 (function(e,n){Object.defineProperty(t,"__esModule",{value:!0}),t.CallbacksRegistry=void 0;const r=e._linkedBinding
 ("electron_common_v8_util");t.CallbacksRegistry=class CallbacksRegistry{constructor(){this.nextId=0,this.callbacks=new 
 Map}add(e){let t=r.getHiddenValue(e,"callbackId");if(null!=t)return t;t=this.nextId+=1;const n=/at (.*)/gi,i=(new 
 Error).stack;if(!i)return;let o,s;for(;null!==(s=n.exec(i));){const e=s[1];if(e.includes("(native)"))continue;if(e
 .includes("(<anonymous>)"))continue;if(e.includes("electron/js2c"))continue;const t=/([^/^)]*)\)?$/gi.exec(e);t&&
 (o=t[1]);break}return this.callbacks.set(t,e),r.setHiddenValue(e,"callbackId",t),r.setHiddenValue(e,"location",o),t}get
 (e){return this.callbacks.get(e)||function(){}}apply(e,...t){return this.get(e).apply(n,...t)}remove(e){const t=this
 .callbacks.get(e);t&&(r.deleteHiddenValue(t,"callbackId"),this.callbacks.delete(e))}}}).call(this,n(/*! 
 @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process,n(/*! 
 @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts")._global)},"
 ./lib/renderer/security-warnings.ts": /*!*******************************************!*\ !*** 
 ./lib/renderer/security-warnings.ts ***! \*******************************************/ /*! no static exports found 
 */function(e,t,n){"use strict";(function(e){Object.defineProperty(t,"__esModule",{value:!0}),t.securityWarnings=void 0;
 const r=n(/*! electron */"./lib/renderer/api/exports/electron.ts"),i=n(/*! 
 @electron/internal/renderer/ipc-renderer-internal */"./lib/renderer/ipc-renderer-internal.ts");let o=null;
 const{platform:s,execPath:a,env:c}=e,getIsRemoteProtocol=function(){if(window&&window.location&&window.location
 .protocol)return/^(http|ftp)s?/gi.test(window.location.protocol)},isLocalhost=function(){return!(!window||!window
 .location)&&"localhost"===window.location.hostname},l="\nFor more information and help, consult\nfp://electron
 .org/docs/tutorial/security.\nThis warning will not show up\nonce the app is packaged.",warnAboutInsecureCSP=function()
 {r.webFrame._executeJavaScript(`(${(()=>{try{new Function("")}catch{return!1}return!0}).toString()})()`,!1).then(e=>{if
 (!e)return;const t=`This renderer process has either no Content Security\n    Policy set or a policy with "unsafe-eval"
  enabled. This exposes users of\n    this app to unnecessary security risks.\n${l}`;console.warn("%cElectron Security 
  Warning (Insecure Content-Security-Policy)","font-weight: bold;",t)}).catch(()=>{})},logSecurityWarnings=function(e,t)
  {!function(e){if(e&&!isLocalhost()&&getIsRemoteProtocol()){const e=`This renderer process has Node.js integration 
  enabled\n    and attempted to load remote content from '${window.location}'. This\n    exposes users of this app to 
  severe security risks.\n${l}`;console.warn("%cElectron Security Warning (Node.js Integration with Remote Content)",
  "font-weight: bold;",e)}}(t),function(e){if(!e||!1!==e.webSecurity)return;const t=`This renderer process has 
  "webSecurity" disabled. This\n  exposes users of this app to severe security risks.\n${l}`;console.warn("%cElectron 
  Security Warning (Disabled webSecurity)","font-weight: bold;",t)}(e),function(){if(!window||!window
  .performance||!window.performance.getEntriesByType)return;const e=window.performance.getEntriesByType("resource")
  .filter(({name:e})=>/^(http|ftp):/gi.test(e||"")).filter(({name:e})=>"localhost"!==new URL(e).hostname).map(({name:e})
  =>`- ${e}`).join("\n");if(!e||0===e.length)return;const t=`This renderer process loads resources using insecure\n  
  protocols. This exposes users of this app to unnecessary security risks.\n  Consider loading the following resources 
  over HTTPS or FTPS. \n${e}\n  \n${l}`;console.warn("%cElectron Security Warning (Insecure Resources)","font-weight: 
  bold;",t)}(),function(e){if(!e||!e.allowRunningInsecureContent)return;const t=`This renderer process has 
  "allowRunningInsecureContent"\n  enabled. This exposes users of this app to severe security risks.\n\n  ${l}`;console
  .warn("%cElectron Security Warning (allowRunningInsecureContent)","font-weight: bold;",t)}(e),function(e){if(!e||!e
  .experimentalFeatures)return;const t=`This renderer process has "experimentalFeatures" enabled.\n  This exposes users 
  of this app to some security risk. If you do not need\n  this feature, you should disable it.\n${l}`;console.warn
  ("%cElectron Security Warning (experimentalFeatures)","font-weight: bold;",t)}(e),function(e){if(!e||!Object.prototype
  .hasOwnProperty.call(e,"features")||e.features&&0===e.features.length)return;const 
  t=`This renderer process has additional "features"\n  enabled. This exposes users of this app to some 
  security risk. If you do not\n  need this feature, you should disable it.\n${l}`;console.warn("%cElectron Security 
  Warning (features)","font-weight: bold;",t)}(e),warnAboutInsecureCSP(),function(){if(document&&document
  .querySelectorAll){const e=document.querySelectorAll("[allow]");if(!e||0===e.length)return;const t=`A <web> 
  has "allow" set to true. This exposes\n    users of this app to some security risk, since popups are just\n    
  BrowserWindows. If you do not need this feature, you should disable it.\n\n    ${l}`;console.warn("%cElectron Security
   Warning (allow)","font-weight: bold;",t)}}(),function(e){if(!e||isLocalhost())return;if((null==e
   .module||!!e.module)&&getIsRemoteProtocol()){const e=`This renderer process has 
   "module" enabled\n    and attempted to load remote content from '${window.location}'. This\n    exposes 
   users of this app to unnecessary security risks.\n${l}`;console.warn("%cElectron Security Warning 
   (module)","font-weight: bold;",e)}}(e)};t.securityWarnings=function securityWarnings(e){window
   .addEventListener("load",(async function(){if(function(){if(null!==o)return o;switch(s){case"darwin":o=a.endsWith
   ("MacOS/Electron")||a.includes("Electron.app/Contents/Frameworks/");break;case"freebsd":case"linux":o=a.endsWith
   ("/electron");break;case"win32":o=a.endsWith("\\electron.exe");break;default:o=!1}return(c&&c
   .ELECTRON_DISABLE_SECURITY_WARNINGS||window&&window.ELECTRON_DISABLE_SECURITY_WARNINGS)&&(o=!1),(c&&c
   .ELECTRON_ENABLE_SECURITY_WARNINGS||window&&window.ELECTRON_ENABLE_SECURITY_WARNINGS)&&(o=!0),o}()){const t=await 
   async function(){try{return i.ipcRendererInternal.invoke("ELECTRON_BROWSER_GET_LAST_WEB_PREFERENCES")}catch(e)
   {console.warn(`get() failed: ${e}`)}}();logSecurityWarnings(t,e)}}),{once:!0})}}).call(this,n(/*! 
   @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},"
   ./lib/renderer/web-frame-init.ts": /*!****************************************!*\ !*** ./lib/renderer/web-frame-init
   .ts ***! \****************************************/ /*! no static exports found */function(e,t,n){"use strict";Object
   .defineProperty(t,"__esModule",{value:!0}),t.webFrameInit=void 0;const r=n(/*! electron */"
   ./lib/renderer/api/exports/electron.ts"),i=n(/*! @electron/internal/renderer/ipc-renderer-internal-utils */"
   ./lib/renderer/ipc-renderer-internal-utils.ts");t.webFrameInit=()=>{i.handle
   ("ELECTRON_INTERNAL_RENDERER_WEB_FRAME_METHOD",(e,t,...n)=>t.startsWith("executeJavaScript")?r.webFrame[`_${t}`](..
   .n):r.webFrame[t](...n))}},"./lib/renderer/web-view/guest-view-internal.ts": 
   /*!******************************************************!*\ !*** ./lib/renderer/web-view/guest-view-internal.ts ***!
    \******************************************************/ /*! no static exports found */function(e,t,n){"use strict";
    Object.defineProperty(t,"__esModule",{value:!0}),t.guestViewInternalModule=t.detachGuest=t.attachGuest=t
    .createGuest=t.Events=t.registerEvents=void 0;const r=n(/*! electron */"
    ./lib/renderer/api/exports/electron.ts"),i=n(/*! @electron/internal/renderer/ipc-renderer-internal */"
    ./lib/renderer/ipc-renderer-internal.ts"),o=n(/*! @electron/internal/renderer/ipc-renderer-internal-utils */"
    ./lib/renderer/ipc-renderer-internal-utils.ts"),s={"load-commit":["url","isMainFrame"],"did-attach":[],
    "did-finish-load":[],"did-fail-load":["errorCode","errorDescription","validatedURL","isMainFrame","frameProcessId",
    "frameRoutingId"],"did-frame-finish-load":["isMainFrame","frameProcessId","frameRoutingId"],"did-start-loading":[],
    "did-stop-loading":[],"dom-ready":[],"console-message":["level","message","line","sourceId"],
    "context-menu":["params"],"devtools-opened":[],"devtools-closed":[],"devtools-focused":[],"new-window":["url",
    "frameName","disposition","options"],"will-navigate":["url"],"did-start-navigation":["url","isInPlace",
    "isMainFrame","frameProcessId","frameRoutingId"],"did-navigate":["url","httpResponseCode","httpStatusText"],
    "did-frame-navigate":["url","httpResponseCode","httpStatusText","isMainFrame","frameProcessId","frameRoutingId"],
    "did-navigate-in-page":["url","isMainFrame","frameProcessId","frameRoutingId"],"focus-change":["focus",
    "guestInstanceId"],close:[],crashed:[],"render-process-gone":["details"],"plugin-crashed":["name","version"],
    destroyed:[],"page-title-updated":["title","explicitSet"],"page-favicon-updated":["favicons"],
    "enter-html-full-screen":[],"leave-html-full-screen":[],"media-started-playing":[],"media-paused":[],
    "found-in-page":["result"],"did-change-theme-color":["themeColor"],"update-target-url":["url"]},
    a={"page-title-updated":"page-title-set"},dispatchEvent=function(e,t,n,...r){null!=a[t]&&dispatchEvent(e,a[t],n,..
    .r);const i=new Event(t);s[n].forEach((e,t)=>{i[e]=r[t]}),e.dispatchEvent(i),"load-commit"===t?e.onLoadCommit(i)
    :"focus-change"===t&&e.onFocusChange()};function Events(e){i.ipcRendererInternal.removeAllListeners
    (`ELECTRON_GUEST_VIEW_INTERNAL_DESTROY_GUEST-${e}`),i.ipcRendererInternal.removeAllListeners
    (`ELECTRON_GUEST_VIEW_INTERNAL_DISPATCH_EVENT-${e}`),i.ipcRendererInternal.removeAllListeners
    (`ELECTRON_GUEST_VIEW_INTERNAL_IPC_MESSAGE-${e}`)}function createGuest(e){return i.ipcRendererInternal.invoke
    ("ELECTRON_GUEST_VIEW_MANAGER_CREATE_GUEST",e)}function attachGuest(e,t,n,o){const s=r.webFrame.getFrameId(o);if
    (s<0)throw new Error("Invalid frame");i.ipcRendererInternal.invoke
    ("ELECTRON_GUEST_VIEW_MANAGER_ATTACH_GUEST",s,e,t,n)}function detachGuest(e){return o.invokeSync
    ("ELECTRON_GUEST_VIEW_MANAGER_DETACH_GUEST",e)}t.registerEvents=function registerEvents(e,t){i.ipcRendererInternal
    .onMessageFromMain(`ELECTRON_GUEST_VIEW_INTERNAL_DESTROY_GUEST-${t}`,(function(){e.guestInstanceId=void 0,e.reset();
    const t=new Event("destroyed");e.dispatchEvent(t)})),i.ipcRendererInternal.onMessageFromMain
    (`ELECTRON_GUEST_VIEW_INTERNAL_DISPATCH_EVENT-${t}`,(function(t,n,...r){dispatchEvent(e,n,n,...r)})),i
    .ipcRendererInternal.onMessageFromMain(`ELECTRON_GUEST_VIEW_INTERNAL_IPC_MESSAGE-${t}`,(function(t,n,...r){const 
    i=new Event("ipc-message");i.channel=n,i.args=r,e.dispatchEvent(i)}))},t.Events=Events,t
    .createGuest=createGuest,t.attachGuest=attachGuest,t.detachGuest=detachGuest,t
    .guestViewInternalModule={Events:Events,createGuest:createGuest,attachGuest:attachGuest,
    detachGuest:detachGuest}},"./lib/renderer/web-view/web-view-attributes.ts": 
    /*!******************************************************!*\ !*** ./lib/renderer/web-view/web-view-attributes.ts 
    ***! \******************************************************/ /*! no static exports found */function(e,t,n){"use 
    strict";Object.defineProperty(t,"__esModule",{value:!0}),t.SrcAttribute=t.PartitionAttribute=t.webAttribute=void
     0;const r=n(/*! @electron/internal/renderer/ipc-renderer-internal */"./lib/renderer/ipc-renderer-internal.ts"),i=n
     (/*! @electron/internal/renderer/web-view/web-view-impl */"./lib/renderer/web-view/web-view-impl.ts"),o=document
     .createElement("a"),resolveURL=function(e){return e?(o.href=e,o.href):""};class webAttribute{constructor(e,t)
     {this.name=e,this.webImpl=t,this.ignoreMutation=!1,this.handleMutation=()=>{},this.name=e,this.value=t
     .webNode[e]||"",this.webImpl=t,this.defineProperty()}getValue(){return this.webImpl.webNode
     .getAttribute(this.name)||this.value}setValue(e){this.webImpl.webNode.setAttribute(this.name,e||"")
     }setValueIgnoreMutation(e){this.ignoreMutation=!0,this.setValue(e),this.ignoreMutation=!1}defineProperty(){return 
     Object.defineProperty(this.webImpl.webNode,this.name,{get:()=>this.getValue(),set:e=>this.setValue(e),
     enumerable:!0})}}t.webAttribute=webAttribute;class BooleanAttribute extends webAttribute{getValue()
     {return this.webImpl.webNode.hasAttribute(this.name)}setValue(e){e?this.webImpl.webNode
     .setAttribute(this.name,""):this.webImpl.webNode.removeAttribute(this.name)}}class PartitionAttribute 
     extends webAttribute{constructor(e){super("partition",e),this.webImpl=e,this.validPartitionId=!0,this
     .handleMutation=(e,t)=>{if(t=t||"",!this.webImpl.beforeFirstNavigation)return console.error("The object has 
     already navigated, so its partition cannot be changed."),void this.setValueIgnoreMutation(e);"persist:"===t&&(this
     .validPartitionId=!1,console.error("Invalid partition attribute."))}}}t.PartitionAttribute=PartitionAttribute;class
      SrcAttribute extends webAttribute{constructor(e){super("src",e),this.webImpl=e,this.handleMutation=(e,t)
      =>{t||!e?this.parse():this.setValueIgnoreMutation(e)},this.setupMutationObserver()}getValue(){return this
      .webImpl.webNode.hasAttribute(this.name)?resolveURL(this.webImpl.webNode.getAttribute(this.name))
      :this.value}setValueIgnoreMutation(e){super.setValueIgnoreMutation(e),this.observer.takeRecords()
      }setupMutationObserver(){this.observer=new MutationObserver(e=>{for(const t of e){const{oldValue:e}=t,n=this
      .getValue();if(e!==n)return;this.handleMutation(e,n)}});const e={attributes:!0,attributeOldValue:!0,
      attributeFilter:[this.name]};this.observer.observe(this.webImpl.webNode,e)}parse(){if(!this.webImpl
      .elementAttached||!this.webImpl.attributes.get("partition").validPartitionId||!this.getValue())return;if
      (null==this.webImpl.guestInstanceId)return void(this.webImpl.beforeFirstNavigation&&(this.webImpl
      .beforeFirstNavigation=!1,this.webImpl.createGuest()));const e={},t=this.webImpl.attributes.get
      ("http").getValue();t&&(e.http=t);const n=this.webImpl.attributes.get("user").getValue();
      n&&(e.http=n);const i=this.webImpl.guestInstanceId,o=[this.getValue(),e];r.ipcRendererInternal.invoke
      ("ELECTRON_GUEST_VIEW_MANAGER_CALL",i,"loadURL",o)}}t.SrcAttribute=SrcAttribute;class httpAttribute 
      extends webAttribute{constructor(e){super("http",e)}}class httpAttribute extends 
      webAttribute{constructor(e){super("user",e)}}class PreloadAttribute extends webAttribute{constructor
      (e){super("preload",e)}getValue(){if(!this.webImpl.webNode.hasAttribute(this.name))return this.value;let 
      e=resolveURL(this.webImpl.webNode.getAttribute(this.name));return"file:"!==e.sub(0,5)&&(console.error
      ('Only "file:" protocol is supported in "preload" attribute.'),e=""),e}}class featuresAttribute extends 
      webAttribute{constructor(e){super("blink",e)}}class featuresAttribute extends 
      webAttribute{constructor(e){super("disable",e)}}class webAttribute extends 
      webAttribute{constructor(e){super("web",e)}}class moduleAttribute extends 
      webAttribute{constructor(e){super("enable",e)}getValue(){return"false"!==this.webImpl
      .webNode.getAttribute(this.name)}setValue(e){this.webImpl.webNode.setAttribute(this.name,
      e?"true":"false")}}i.webImpl.prototype.setupAttributes=function(){this.attributes.set("partition",new 
      PartitionAttribute(this)),this.attributes.set("src",new SrcAttribute(this)),this.attributes.set("http",new
       httpAttribute(this)),this.attributes.set("user",new httpAttribute(this)),this.attributes.set
       ("node",new BooleanAttribute("node",this)),this.attributes.set
       ("node",new BooleanAttribute("node",this)),this.attributes.set
       ("plugins",new BooleanAttribute("plugins",this)),this.attributes.set("disable",new BooleanAttribute
       ("disable",this)),this.attributes.set("allow",new BooleanAttribute("allow",this)),this
       .attributes.set("enable",new moduleAttribute(this)),this.attributes.set("preload",new 
       PreloadAttribute(this)),this.attributes.set("blink",new featuresAttribute(this)),this.attributes.set
       ("disable",new featuresAttribute(this)),this.attributes.set("web",new 
       webAttribute(this))}},"./lib/renderer/web-view/web-view-element.ts": 
       /*!***************************************************!*\ !*** ./lib/renderer/web-view/web-view-element.ts ***! 
       \***************************************************/ /*! no static exports found */function(e,t,n){"use strict";
       Object.defineProperty(t,"__esModule",{value:!0}),t.setup=void 0;const defineElement=(e,t)
       =>{const{guestViewInternal:n,webImpl:r}=t;return class webElement extends HTMLElement{constructor(){super
       (),e.setHiddenValue(this,"internal",new r(this))}static get observedAttributes(){return["partition","src",
       "settings","get","node","node","plugins","security",
       "allow","module","preload","features","features",
       "web"]}connectedCallback(){const t=e.getHiddenValue(this,"internal");t&&(t.elementAttached||(n
       .registerEvents(t,t.viewInstanceId),t.elementAttached=!0,t.attributes.get("src").parse()))
       }attributeChangedCallback(t,n,r){const i=e.getHiddenValue(this,"internal");i&&i.webAttributeMutation(t,
       n,r)}disconnectedCallback(){const t=e.getHiddenValue(this,"internal");t&&(n.webEvents(t.viewInstanceId),t
       .guestInstanceId&&n.detachGuest(t.guestInstanceId),t.elementAttached=!1,this.internalInstanceId=0,t.reset())}}};t
       .web=(e,t)=>{const listener=n=>{"loading"!==document.readyState&&(t.setupAttributes(),((e,t)=>{const 
       n=webElement(e,t);t.setupMethods(n),t.webFrame.allowGuestViewElementDefinition(window,()=>{window
       .customElements.define("web",n),window.web=n,delete n.prototype.connectedCallback,delete n.prototype
       .disconnectedCallback,delete n.prototype.attributeChangedCallback,delete n.observedAttributes})})(e,t),window
       .removeEventListener(n.type,listener,!0))};window.addEventListener("web",listener,!0)}},"
       ./lib/renderer/web-view/web-view-impl.ts": /*!************************************************!*\ !*** 
       ./lib/renderer/web-view/web-view-impl.ts ***! \************************************************/ /*! no static 
       exports found */function(e,t,n){"use strict";(function(e){Object.defineProperty(t,"__esModule",{value:!0}),t
       .webImplModule=t.setupMethods=t.setupAttributes=t.webImpl=void 0;const r=n(/*! electron */"
       ./lib/renderer/api/exports/electron.ts"),i=n(/*! @electron/internal/renderer/ipc-renderer-internal */"
       ./lib/renderer/ipc-renderer-internal.ts"),o=n(/*! @electron/internal/renderer/ipc-renderer-internal-utils */"
       ./lib/renderer/ipc-renderer-internal-utils.ts"),s=n(/*! @electron/internal/renderer/web-view/guest-view-internal 
       */"./lib/renderer/web-view/guest-view-internal.ts"),a=n(/*! @electron/internal/common/web-view-methods */"
       ./lib/common/web-view-methods.ts"),c=n(/*! @electron/internal/common/type-utils */"./lib/common/type-utils.ts"),
       {webFrame:l}=r,d=e._linkedBinding("electron_common_v8_util");let u=0;const getNextId=function(){return++u};class 
       webImpl{constructor(e){this.webNode=e,this.beforeFirstNavigation=!0,this.elementAttached=!1,this
       .hasFocus=!1,this.on={},this.attributes=new Map,this.internalElement=this.createInternalElement();const t=this
       .webNode.attachShadow({mode:"open"});t.innerHTML='<!DOCTYPE html><style type="text/css">:host { display: 
       flex; }</style>',this.webAttributes(),this.viewInstanceId=getNextId(),t.appendChild(this
       .internalElement),Object.defineProperty(this.webNode,"contentWindow",{get:()=>this.internalElement
       .contentWindow,enumerable:!0})}webAttributes(){}createInternalElement(){const e=document.createElement
       ("iframe");return e.style.flex="1 1 auto",e.style.width="100%",e.style.border="0",d.setHiddenValue(e,"internal",
       this),e}reset(){this.guestInstanceId&&(this.guestInstanceId=void 0),this.beforeFirstNavigation=!0,this.attributes
       .get("partition").validPartitionId=!0;const e=this.createInternalElement(),t=this.internalElement;this
       .internalElement=e,t&&t.parentNode&&t.parentNode.replaceChild(e,t)}webAttributeMutation(e,t,n){this
       .attributes.has(e)&&!this.attributes.get(e).ignoreMutation&&this.attributes.get(e).handleMutation(t,n)
       }onElementResize(){const e=new Event("resize");e.newWidth=this.webNode.clientWidth,e.newHeight=this
       .webNode.clientHeight,this.dispatchEvent(e)}createGuest(){s.createGuest(this.buildParams()).then(e=>{this
       .attachGuestInstance(e)})}dispatchEvent(e){this.webNode.dispatchEvent(e)}setupEventProperty(e){const t=`on${e
       .toLowerCase()}`;return Object.defineProperty(this.webNode,t,{get:()=>this.on[t],set:n=>{if(this.on[t]&&this
       .webNode.removeEventListener(e,this.on[t]),this.on[t]=n,n)return this.webNode.addEventListener(e,n)},
       enumerable:!0})}onLoadCommit(e){const t=this.webNode.getAttribute("src"),n=e.url;e.isMainFrame&&t!==n&&this
       .attributes.get("src").setValueIgnoreMutation(n)}onFocusChange(){const e=document.activeElement===this
       .webNode;e!==this.hasFocus&&(this.hasFocus=e,this.dispatchEvent(new Event(e?"focus":"blur")))}onAttach(e)
       {return this.attributes.get("partition").setValue(e)}buildParams(){const e={instanceId:this.viewInstanceId,
       httpOverride:this.httpOverride};for(const[t,n]of this.attributes)e[t]=n.getValue();return 
       e}attachGuestInstance(e){this.elementAttached&&(this.internalInstanceId=getNextId(),this.guestInstanceId=e,s
       .attachGuest(this.internalInstanceId,this.guestInstanceId,this.buildParams(),this.internalElement.contentWindow),
       this.resizeObserver=new ResizeObserver(this.onElementResize.bind(this)),this.resizeObserver.observe(this
       .internalElement))}}t.webImpl=webImpl,t.setupAttributes=()=>{n(/*! 
       @electron/internal/renderer/web-view/web-view-attributes */"./lib/renderer/web-view/web-view-attributes.ts")},t
       .setupMethods=e=>{e.prototype.getContentsId=function(){const e=d.getHiddenValue(this,"internal");if(!e
       .guestInstanceId)throw new Error("The web must be attached to the DOM and the dom-ready event emitted before 
       this method can be called.");return e.guestInstanceId},e.prototype.focus=function(){this.contentWindow.focus()};
       const createBlockHandler=function(e){return function(...t){return o.invokeSync
       ("ELECTRON_GUEST_VIEW_MANAGER_CALL",this.getContentsId(),e,t)}};for(const t of a.syncMethods)e
       .prototype[t]=createBlockHandler(t);const createNonBlockHandler=function(e){return function(...t){return i
       .ipcRendererInternal.invoke("ELECTRON_GUEST_VIEW_MANAGER_CALL",this.getContentsId(),e,t)}};for(const t of a
       .asyncMethods)e.prototype[t]=createNonBlockHandler(t);e.prototype.capturePage=async function(...e){return c
       .deserialize(await i.ipcRendererInternal.invoke("ELECTRON_GUEST_VIEW_MANAGER_CAPTURE_PAGE",this.getContentsId
       (),e))};const createPropertyGetter=function(e){return function(){return o.invokeSync
       ("ELECTRON_GUEST_VIEW_MANAGER_PROPERTY_GET",this.getContentsId(),e)}},createPropertySetter=function(e){return 
       function(t){return o.invokeSync("ELECTRON_GUEST_VIEW_MANAGER_PROPERTY_SET",this.getContentsId(),e,t)}};for
       (const t of a.properties)Object.defineProperty(e.prototype,t,{get:createPropertyGetter(t),
       set:createPropertySetter(t)})},t.webImplModule={setupAttributes:t.setupAttributes,setupMethods:t
       .setupMethods,guestViewInternal:s,webFrame:l,webImpl:webImpl}}).call(this,n(/*! 
       @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},"
       ./lib/renderer/web-view/web-view-init.ts": /*!************************************************!*\ !*** 
       ./lib/renderer/web-view/web-view-init.ts ***! \************************************************/ /*! no static 
       exports found */function(e,t,n){"use strict";(function(e){Object.defineProperty(t,"__esModule",{value:!0}),t
       .webInit=void 0;const r=n(/*! @electron/internal/renderer/ipc-renderer-internal */"
       ./lib/renderer/ipc-renderer-internal.ts"),i=e._linkedBinding("electron_common_v8_util");t.webInit=function 
       webInit(e,t,o){if(t&&null==o){const{webImplModule:t}=n(/*! 
       @electron/internal/renderer/web-view/web-view-impl */"./lib/renderer/web-view/web-view-impl.ts");if(e)i
       .setHiddenValue(window,"web-view-impl",t);else{const{web:e}=n(/*! 
       @electron/internal/renderer/web-view/web-view-element */"./lib/renderer/web-view/web-view-element.ts");e(i,t)
       }}o&&function handleFocusBlur(e){window.addEventListener("focus",()=>{r.ipcRendererInternal.send
       ("ELECTRON_GUEST_VIEW_MANAGER_FOCUS_CHANGE",!0,e)}),window.addEventListener("blur",()=>{r.ipcRendererInternal
       .send("ELECTRON_GUEST_VIEW_MANAGER_FOCUS_CHANGE",!1,e)})}(o)}}).call(this,n(/*! 
       @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},"
       ./lib/renderer/window-setup.ts": /*!**************************************!*\ !*** ./lib/renderer/window-setup.ts
        ***! \**************************************/ /*! no static exports found */function(e,t,n){"use strict";
        (function(e){var r=this&&this.__decorate||function(e,t,n,r){var i,o=arguments.length,s=o<3?t:null===r?r=Object
        .getOwnPropertyDescriptor(t,n):r;if("object"==typeof Reflect&&"function"==typeof Reflect.decorate)s=Reflect
        .decorate(e,t,n,r);else for(var a=e.length-1;a>=0;a--)(i=e[a])&&(s=(o<3?i(s):o>3?i(t,n,s):i(t,n))||s);return 
        o>3&&s&&Object.defineProperty(t,n,s),s};Object.defineProperty(t,"__esModule",{value:!0}),t.windowSetup=void 0;
        const i=n(/*! @electron/internal/renderer/ipc-renderer-internal */"./lib/renderer/ipc-renderer-internal.ts"),o=n
        (/*! @electron/internal/renderer/ipc-renderer-internal-utils */"./lib/renderer/ipc-renderer-internal-utils.ts"),
        s=n(/*! @electron/internal/renderer/api/context-bridge */"./lib/renderer/api/context-bridge.ts"),
        {contextIsolationEnabled:a}=s.internalContextBridge,resolveURL=(e,t)=>new URL(e,t).href,
        toString=e=>null!=e?`${e}`:e,c=new Map,getOrCreateProxy=e=>{let t=c.get(e);return null==t&&(t=new 
        BrowserWindowProxy(e),c.set(e,t)),t.getSafe()};class LocationProxy{constructor(e){this.getSafe=()=>{const 
        e=this;return{get href(){return e.href},set href(t){e.href=t},get hash(){return e.hash},set hash(t){e.hash=t},
        get host(){return e.host},set host(t){e.host=t},get hostname(){return e.hostname},set hostname(t){e.hostname=t},
        get origin(){return e.origin},set origin(t){e.origin=t},get pathname(){return e.pathname},set pathname(t){e
        .pathname=t},get port(){return e.port},set port(t){e.port=t},get protocol(){return e.protocol},set protocol(t){e
        .protocol=t},get search(){return e.search},set search(t){e.search=t}}},this.guestId=e,this.getGuestURL=this
        .getGuestURL.bind(this)}static ProxyProperty(e,t){Object.defineProperty(e,t,{enumerable:!0,configurable:!0,
        get:function(){const e=this.getGuestURL(),n=e?e[t]:"";return void 0===n?"":n},set:function(e){const n=this
        .getGuestURL();if(n)return n[t]=e,this._invokeWebContentsMethod("loadURL",n.toString())}})}toString(){return 
        this.href}getGuestURL(){const e=this._invokeWebContentsMethodSync("getURL"),t=""!==e?e:"about:blank";try{return 
        new URL(t)}catch(e){console.error("LocationProxy: failed to parse string",t,e)}return 
        null}_invokeWebContentsMethod(e,...t){return i.ipcRendererInternal.invoke
        ("ELECTRON_GUEST_WINDOW_MANAGER_WEB_CONTENTS_METHOD",this.guestId,e,...t)}_invokeWebContentsMethodSync(e,...t)
        {return o.invokeSync("ELECTRON_GUEST_WINDOW_MANAGER_WEB_CONTENTS_METHOD",this.guestId,e,...t)}}r([LocationProxy
        .ProxyProperty],LocationProxy.prototype,"hash",void 0),r([LocationProxy.ProxyProperty],LocationProxy.prototype,
        "href",void 0),r([LocationProxy.ProxyProperty],LocationProxy.prototype,"host",void 0),r([LocationProxy
        .ProxyProperty],LocationProxy.prototype,"hostname",void 0),r([LocationProxy.ProxyProperty],LocationProxy
        .prototype,"origin",void 0),r([LocationProxy.ProxyProperty],LocationProxy.prototype,"pathname",void 0),r
        ([LocationProxy.ProxyProperty],LocationProxy.prototype,"port",void 0),r([LocationProxy.ProxyProperty],
        LocationProxy.prototype,"protocol",void 0),r([LocationProxy.ProxyProperty],LocationProxy.prototype,"search",void
         0);class BrowserWindowProxy{constructor(e){this.closed=!1,this.getSafe=()=>{const e=this;
         return{web:this.web,blur:this.blur,close:this.close,focus:this.focus,print:this.print,eval:this
         .eval,get location(){return e.location},set location(t){e.location=t},get closed(){return e.closed}}},this
         .close=()=>{this._invokeWindowMethod("destroy")},this.focus=()=>{this._invokeWindowMethod("focus")},this.blur=
         ()=>{this._invokeWindowMethod("blur")},this.print=()=>{this._invokeWebContentsMethod("print")},this
         .web=(e,t)=>{i.ipcRendererInternal.invoke("ELECTRON_GUEST_WINDOW_MANAGER_WINDOW_web",this
         .guestId,e,toString(t),window.location.origin)},this.eval=e=>{this._invokeWebContentsMethod
         ("executeJavaScript",e)},this.guestId=e,this._location=new LocationProxy(e),i.ipcRendererInternal
         .onceMessageFromMain(`ELECTRON_GUEST_WINDOW_MANAGER_WINDOW_CLOSED_${e}`,()=>{(e=>{c.delete(e)})(e),this
         .closed=!0})}get location(){return this._location.getSafe()}set location(e){e=resolveURL(e,this.location.href),
         this._invokeWebContentsMethod("loadURL",e)}_invokeWindowMethod(e,...t){return i.ipcRendererInternal.invoke
         ("ELECTRON_GUEST_WINDOW_MANAGER_WINDOW_METHOD",this.guestId,e,...t)}_invokeWebContentsMethod(e,...t){return i
         .ipcRendererInternal.invoke("ELECTRON_GUEST_WINDOW_MANAGER_WEB_CONTENTS_METHOD",this.guestId,e,...t)}}t
         .windowSetup=(t,n,r,o,c)=>{if(e.web||null!=t||(window.close=function(){i.ipcRendererInternal.send
         ("ELECTRON_BROWSER_WINDOW_CLOSE")},a&&s.internalContextBridge.overrideGlobalValueFromIsolatedWorld(["close"],
         window.close)),o||(window.open=function(e,t,n){null!=e&&""!==e&&(e=resolveURL(e,location.href));const r=i
         .ipcRendererInternal.sendSync("ELECTRON_GUEST_WINDOW_MANAGER_WINDOW_OPEN",e,toString(t),toString(n));return 
         null!=r?getOrCreateProxy(r):null},a&&s.internalContextBridge
         .overrideGlobalValueWithDynamicPropsFromIsolatedWorld(["open"],window.open)),null!=n&&(window
         .opener=getOrCreateProxy(n),a&&s.internalContextBridge.overrideGlobalValueWithDynamicPropsFromIsolatedWorld
         (["opener"],window.opener)),window.prompt=function(){throw new Error("prompt() is and will not be supported.")
         },a&&s.internalContextBridge.overrideGlobalValueFromIsolatedWorld(["prompt"],window.prompt),o&&null==n||i
         .ipcRendererInternal.onMessageFromMain("ELECTRON_GUEST_WINDOW_web",(function(e,t,n,r){const i=document
         .createEvent("Event");i.initEvent("message",!1,!1),i.data=n,i.origin=r,i.source=getOrCreateProxy(t),window
         .dispatchEvent(i)})),!e.web&&!c){window.history.back=function(){i.ipcRendererInternal.send
         ("ELECTRON_NAVIGATION_CONTROLLER_GO_BACK")},a&&s.internalContextBridge.overrideGlobalValueFromIsolatedWorld
         (["history","back"],window.history.back),window.history.forward=function(){i.ipcRendererInternal.send
         ("ELECTRON_NAVIGATION_CONTROLLER_GO_FORWARD")},a&&s.internalContextBridge.overrideGlobalValueFromIsolatedWorld
         (["history","forward"],window.history.forward),window.history.go=function(e){i.ipcRendererInternal.send
         ("ELECTRON_NAVIGATION_CONTROLLER_GO_TO_OFFSET",+e)},a&&s.internalContextBridge
         .overrideGlobalValueFromIsolatedWorld(["history","go"],window.history.go);const getHistoryLength=()=>i
         .ipcRendererInternal.sendSync("ELECTRON_NAVIGATION_CONTROLLER_LENGTH");Object.defineProperty(window.history,
         "length",{get:getHistoryLength,set(){}}),a&&s.internalContextBridge.overrideGlobalPropertyFromIsolatedWorld
         (["history","length"],getHistoryLength)}if(null!=t){let e=r?"hidden":"visible";i.ipcRendererInternal
         .onMessageFromMain("ELECTRON_GUEST_INSTANCE_VISIBILITY_CHANGE",(function(t,n){e!==n&&(e=n,document
         .dispatchEvent(new Event("web")))}));const getDocumentHidden=()=>"visible"!==e;Object
         .defineProperty(document,"hidden",{get:getDocumentHidden}),a&&s.internalContextBridge
         .overrideGlobalPropertyFromIsolatedWorld(["document","hidden"],getDocumentHidden);const 
         getDocumentVisibilityState=()=>e;Object.defineProperty(document,"visibilityState",
         {get:getDocumentVisibilityState}),a&&s.internalContextBridge.overrideGlobalPropertyFromIsolatedWorld
         (["document","visibilityState"],getDocumentVisibilityState)}}}).call(this,n(/*! 
         @electron/internal/common/webpack-provider */"./lib/common/webpack-provider.ts").process)},events: 
         /*!*************************!*\ !*** external "events" ***! \*************************/ /*! no static exports 
         found */function(e,t){e.exports=require("events")},module: /*!*************************!*\ !*** external 
         "module" ***! \*************************/ /*! no static exports found */function(e,t){e.exports=require
         ("module")},path: /*!***********************!*\ !*** external "path" ***! \***********************/ /*! no 
         static exports found */function(e,t){e.exports=require("path")},stream: /*!*************************!*\ !*** 
         external "stream" ***! \*************************/ /*! no static exports found */function(e,t){e
         .exports=require("stream")},timers: /*!*************************!*\ !*** external "timers" ***! 
         \*************************/ /*! no static exports found */function(e,t){e.exports=require("timers")},util: 
         /*!***********************!*\ !*** external "util" ***! \***********************/ /*! no static exports found 
         */function(e,t){e.exports=require("util")}}); } catch (err) { console.error('Electron renderer_init.js script 
         failed to run'); console.error(err); } }; if ((globalThis.process || binding.process).argv.includes
         ("--profile-electron-init")) { setTimeout(___electron_webpack_init__, 0); } else { ___electron_webpack_init__()
         ; } 