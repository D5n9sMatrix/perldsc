(function s(e){let t;t="string"===e?new String(""):"number"===e?new Number(0):"bigint"===e?Object(BigInt(0))
:"boolean"===e?new Boolean(!1):this;const n=[];try{for(let i=t;i;i=Object.getPrototypeOf(i)){if(
("array"===e||"t"===e)&&i===t&&i.length>9999)continue;const s={items:[],__proto__:null};try{"object"==typeof
i&&Object.prototype.hasOwnProperty.call(i,"constructor")&&i.constructor&&i.constructor.name&&(s.title=i.constructor
.name)}catch(e){}n[n.length]=s;const r=Object.getOwnPropertyNames(i),o=Array.isArray(i);for(let e=0;e<r.length&&s.items
.length<1e4;++e)o&&/^[0-9]/.test(r[e])||(s.items[s.items.length]=r[e])}}catch(e){}return n})