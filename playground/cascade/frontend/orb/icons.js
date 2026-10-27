(()=>{var u={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor","stroke-width":2,"stroke-linecap":"round","stroke-linejoin":"round"};var h=([e,a,o])=>{let r=document.createElementNS("http://www.w3.org/2000/svg",e);return Object.keys(a).forEach(t=>{r.setAttribute(t,String(a[t]))}),o?.length&&o.forEach(t=>{let s=h(t);r.appendChild(s)}),r},S=(e,a={})=>{let r={...u,...a};return h(["svg",r,e])};var g=(...e)=>e.filter((a,o,r)=>!!a&&a.trim()!==""&&r.indexOf(a)===o).join(" ").trim();var k=e=>{for(let a in e)if(a.startsWith("aria-")||a==="role"||a==="title")return!0;return!1};var w=e=>{let a="",o=!1;for(let r of e){if(r==="-"||r==="_"||r<=" "){o=a.length>0;continue}a.length===0?a+=r.toLowerCase():a+=o?r.toUpperCase():r,o=!1}return a};var P=e=>{let a=w(e);return a.charAt(0).toUpperCase()+a.slice(1)};var L=e=>Array.from(e.attributes).reduce((a,o)=>(a[o.name]=o.value,a),{}),A=e=>typeof e=="string"?e:!e||!e.class?"":e.class&&typeof e.class=="string"?e.class.split(" "):e.class&&Array.isArray(e.class)?e.class:"",d=(e,{nameAttr:a,icons:o,attrs:r})=>{let t=e.getAttribute(a);if(t==null)return;let s=P(t),f=o[s];if(!f)return console.warn(`${e.outerHTML} icon name was not found in the provided icons object.`);let l=L(e),B=k(l)?{}:{"aria-hidden":"true"},c={...u,"data-lucide":t,...B,...r,...l},M=A(l),D=A(r),C=g("lucide",`lucide-${t}`,...M,...D);C&&Object.assign(c,{class:C});let F=S(f,c);return e.parentNode?.replaceChild(F,e)};var p=[["path",{d:"M16 5h6"}],["path",{d:"M19 2v6"}],["path",{d:"M21 11.5V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7.5"}],["path",{d:"m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"}],["circle",{cx:"9",cy:"9",r:"2"}]];var m=[["path",{d:"M12 19v3"}],["path",{d:"M19 10v2a7 7 0 0 1-14 0v-2"}],["rect",{x:"9",y:"2",width:"6",height:"13",rx:"3"}]];var x=[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2"}]];var i=[["path",{d:"M18 6 6 18"}],["path",{d:"m6 6 12 12"}]];var n=({icons:e={},nameAttr:a="data-lucide",attrs:o={},root:r=document,inTemplates:t}={})=>{if(!Object.values(e).length)throw new Error(`Please provide an icons object.
If you want to use all the icons you can import it like:
 \`import { createIcons, icons } from 'lucide';
lucide.createIcons({icons});\``);if(typeof r>"u")throw new Error("`createIcons()` only works in a browser environment.");if(Array.from(r.querySelectorAll(`[${a}]`)).forEach(f=>d(f,{nameAttr:a,icons:e,attrs:o})),t&&Array.from(r.querySelectorAll("template")).forEach(l=>n({icons:e,nameAttr:a,attrs:o,root:l.content,inTemplates:t})),a==="data-lucide"){let f=r.querySelectorAll("[icon-name]");f.length>0&&(console.warn("[Lucide] Some icons were found with the now deprecated icon-name attribute. These will still be replaced for backwards compatibility, but will no longer be supported in v1.0 and you should switch to data-lucide"),Array.from(f).forEach(l=>d(l,{nameAttr:"icon-name",icons:e,attrs:o})))}};n({icons:{ImagePlus:p,X:i,Mic:m,Square:x}});})();
/*! Bundled license information:

lucide/dist/esm/defaultAttributes.mjs:
lucide/dist/esm/createElement.mjs:
lucide/dist/esm/shared/src/utils/mergeClasses.mjs:
lucide/dist/esm/shared/src/utils/hasA11yProp.mjs:
lucide/dist/esm/shared/src/utils/toCamelCase.mjs:
lucide/dist/esm/shared/src/utils/toPascalCase.mjs:
lucide/dist/esm/replaceElement.mjs:
lucide/dist/esm/icons/image-plus.mjs:
lucide/dist/esm/icons/mic.mjs:
lucide/dist/esm/icons/square.mjs:
lucide/dist/esm/icons/x.mjs:
lucide/dist/esm/lucide.mjs:
  (**
   * @license lucide v1.61.0 - ISC
   *
   * This source code is licensed under the ISC license.
   * See the LICENSE file in the root directory of this source tree.
   *)
*/
