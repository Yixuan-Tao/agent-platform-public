
var emcData = ___JSON_DATA___;
var overlayConfig = ___OVERLAY_IMAGE___;
var restoreState = window.__EMC_RESTORE_STATE__ || null;
if (restoreState && typeof restoreState.signature !== 'string') restoreState = null;
// 频段分组默认配置（Python 端注入；调用方未在 chart_config.frequency_groups 覆盖时使用）
var DEFAULT_FREQUENCY_GROUPS = ___DEFAULT_FREQUENCY_GROUPS___;

const COLORS = {
    'Class3': '#e74c3c', 'Class4': '#f39c12', 'Class5': '#27ae60',
    'ClassA': '#3498db', 'ClassB': '#9b59b6', 'ClassC': '#1abc9c',
    'Class1': '#e67e22', 'Class2': '#2ecc71', 'Default': '#3498db'
};
const DATASET_PALETTE = [
    '#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c',
    '#e67e22', '#34495e', '#16a085', '#d35400', '#8e44ad', '#2c3e50',
];
const LINE_STYLES = { PK: 'solid', QP: 'dashed', AV: 'dotted' };

function isFlat(limits) {
    if (!limits || typeof limits !== 'object') return false;
    return Object.prototype.hasOwnProperty.call(limits, 'value') || Object.prototype.hasOwnProperty.call(limits, 'formula');
}

function getClasses(limits) {
    if (!limits || typeof limits !== 'object') return [];
    if (isFlat(limits)) return ['Default'];
    return Object.keys(limits).filter(k => limits[k] != null && typeof limits[k] === 'object');
}

function getInfo(limits, cls) {
    if (isFlat(limits)) return limits;
    return limits[cls];
}

function escapeHtml(s) { return String(s == null ? '' : s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\"/g,'&quot;').replace(/'/g,'&#39;'); }
function displayValue(value) { return value == null || value === '' ? '-' : value; }
function formatFrequencyTick(value) {
    const v = Number(value);
    if (!isFinite(v)) return '';
    if (v >= 1000) return (v / 1000).toFixed(v >= 10000 ? 0 : 1).replace(/\\.0$/, '') + ' GHz';
    if (v >= 1) return v.toFixed(v >= 10 ? 0 : 2).replace(/\\.?0+$/, '') + ' MHz';
    if (v >= 0.001) return (v * 1000).toFixed(v >= 0.1 ? 0 : 1).replace(/\\.0$/, '') + ' kHz';
    return (v * 1000000).toFixed(0) + ' Hz';
}

function emitChartState(detail) {
    if (!detail || typeof detail !== 'object') return;
    window.dispatchEvent(new CustomEvent('emc:chart-state-change', { detail: detail }));
}

function controlStateKey(element) {
    return [element.dataset.kind || '', element.dataset.dataset || '', element.dataset.value || ''].join('::');
}

function collectTypeAFilters(panel) {
    const filters = {};
    if (!panel) return filters;
    panel.querySelectorAll('input[type="checkbox"][data-kind]').forEach(function(element){
        filters[controlStateKey(element)] = Boolean(element.checked);
    });
    return filters;
}

function restoreTypeAFilters(panel, filters) {
    if (!panel || !filters || typeof filters !== 'object') return;
    panel.querySelectorAll('input[type="checkbox"][data-kind]').forEach(function(element){
        const key = controlStateKey(element);
        if (typeof filters[key] === 'boolean') element.checked = filters[key];
    });
}

function emitTypeAState(panel) {
    const range = window._typeARange;
    if (!range || range.length !== 2) return;
    emitChartState({
        type:'typeA',
        filters:collectTypeAFilters(panel),
        startValue:range[0],
        endValue:range[1]
    });
}

function visibleTypeASeries(series) {
    return series.filter(function(s){ return s.visible !== false; });
}

function typeARangeFromSeries(series) {
    const visible = visibleTypeASeries(series);
    if (!visible.length) return null;
    const starts = visible.map(function(s){ return Number(s.bandStart); }).filter(function(v){ return isFinite(v) && v > 0; });
    const ends = visible.map(function(s){ return Number(s.bandEnd); }).filter(function(v){ return isFinite(v) && v > 0; });
    return starts.length && ends.length ? [Math.min.apply(null, starts), Math.max.apply(null, ends)] : null;
}

function applyTypeARange(chart, range) {
    if (!chart || !range || range[0] <= 0 || range[1] < range[0]) return;
    window._typeARange = range.slice();
    chart.dispatchAction({ type:'dataZoom', startValue:range[0], endValue:range[1] });
}

function fitTypeAY(chart, series) {
    const range = window._typeARange || typeARangeFromSeries(series);
    const values = [];
    visibleTypeASeries(series).forEach(function(s){
        (s.data || []).forEach(function(point){
            const x = Number(point[0]); const y = Number(point[1]);
            if (isFinite(x) && isFinite(y) && (!range || (x >= range[0] && x <= range[1]))) values.push(y);
        });
    });
    if (!values.length) return;
    const minValue = Math.min.apply(null, values); const maxValue = Math.max.apply(null, values);
    const span = maxValue - minValue;
    const pad = Math.max(5, span * 0.1);
    chart.setOption({ yAxis: { min: Math.floor((minValue-pad)/5)*5, max: Math.ceil((maxValue+pad)/5)*5 } });
}

function constantValue(info) {
    if (!info) return null;
    if (info.value != null) {
        const y = Number(info.value);
        return isFinite(y) ? y : null;
    }
    return null;
}

function parseFrequency(freq) {
    if (!freq) return null;
    if (typeof freq === 'object') return { start: freq.start, end: freq.end };
    const m = String(freq).match(/([\\d.]+)\\s*[-~]\\s*([\\d.]+)/);
    return m ? { start: parseFloat(m[1]), end: parseFloat(m[2]) } : null;
}

function buildSeriesForDataset(dataset, datasetName, datasetIdx) {
    const series = [];
    const bands = dataset && dataset.bands || [];
    const datasetColor = DATASET_PALETTE[datasetIdx % DATASET_PALETTE.length];
    bands.forEach(function(band) {
        if (!band.limits) return;
        const freqObj = parseFrequency(band.frequency);
        const bandRange = freqObj ? (String(freqObj.start) + ' - ' + String(freqObj.end) + ' MHz') : '';
        const bandLabel = band.raw_label || band.service || bandRange;
        const dets = Object.keys(band.limits).filter(d=> band.limits[d] != null);
        dets.forEach(function(det) {
            const limits = band.limits[det];
            const classes = getClasses(limits);
            classes.forEach(function(cls) {
                const info = getInfo(limits, cls);
                if (!info) return;
                const hasFormula = info.formula && info.formula !== '';
                const hasValue = info.value != null && !isNaN(Number(info.value));
                if (!hasFormula && !hasValue) return;
                if (!freqObj) return;
                const f1 = freqObj.start;
                const f2 = freqObj.end;
                if (!f1 || !f2 || f1 <= 0 || f2 <= 0) return;
                let points = Array.isArray(info.points) ? info.points.filter(function(p){ return Array.isArray(p) && p.length === 2 && isFinite(Number(p[0])) && isFinite(Number(p[1])); }) : [];
                if (points.length === 0) {
                    const y = constantValue(info);
                    if (y != null) points = [[f1, y], [f2, y]];
                }
                if (points.length===0) return;
                const shortBand = bandLabel || (bandRange ? bandRange.replace(/\\s*MHz\\s*$/,'') : '');
                const name = datasetName + (shortBand ? ' / ' + shortBand : '') + ' / ' + det;
                const baseColor = datasetColor;
                const styleType = LINE_STYLES[det] || 'solid';
                const width = (det==='PK'?2.5:1.8);
                // emphasis 增强：加粗 + 沿用系列颜色的发光阴影，让悬停曲线在多线图中更突出
                // 透传 band.bw（测量带宽字符串）用于带宽对比面板
                series.push({ name: name, datasetName: datasetName, datasetIdx: datasetIdx, bandId: band.id||'', bandLabel: bandLabel, bandRange: bandRange, bandStart: f1, bandEnd: f2, cls: cls, det: det, unit: info.unit||'', bw: band.bw || null, visible: true, type: 'line', data: points, symbol:'none', smooth:false, lineStyle:{ color: baseColor, type: styleType, width: width }, emphasis:{ focus:'series', lineStyle:{ width: width + 2.5, shadowBlur: 12, shadowColor: baseColor } }, tooltip: { show: true }, label: { show: false } });
            });
        });
    });
    return series;
}

function buildAllSeries(dataObj) {
    const all = [];
    const source = dataObj || emcData;
    if (source && Array.isArray(source.datasets)) {
        source.datasets.forEach(function(ds, idx){
            const name = ds.name || ds.Name || ('dataset'+(idx+1));
            const dObj = ds.data || ds.Data || {};
            const s = buildSeriesForDataset(dObj, name, idx);
            s.forEach(x=> all.push(x));
        });
    } else {
        const name = (source && source.metadata && source.metadata.standard) || 'Dataset';
        const s = buildSeriesForDataset(source, name, 0); s.forEach(x=> all.push(x));
    }
    return all;
}

function getSeriesMeta(series, index){ const s = series[index] || {}; return { name: s.name||'', datasetName: s.datasetName||'Dataset', cls: s.cls||'Unknown', det: s.det||'', bandId: s.bandId||'', label: s.bandLabel || s.bandRange || s.name || '' }; }

function getSelectionMaps(panel){
    const datasetEnabled = {};
    const classEnabled = {};
    const bandEnabled = {};
    const detEnabled = {};
    if (!panel) return { datasetEnabled, classEnabled, bandEnabled, detEnabled };
    panel.querySelectorAll('input[type="checkbox"][data-kind="dataset"]').forEach(el=> datasetEnabled[el.dataset.value]=el.checked);
    panel.querySelectorAll('input[type="checkbox"][data-kind="class"]').forEach(el=>{ const key = el.dataset.value + '||' + (el.dataset.dataset || ''); classEnabled[key]=el.checked; });
    panel.querySelectorAll('input[type="checkbox"][data-kind="band"]').forEach(el=>{ const key = el.dataset.value + '||' + (el.dataset.dataset || ''); bandEnabled[key]=el.checked; });
    panel.querySelectorAll('input[type="checkbox"][data-kind="det"]').forEach(el=>{ const key = el.dataset.value + '||' + (el.dataset.dataset || ''); detEnabled[key]=el.checked; });
    return { datasetEnabled, classEnabled, bandEnabled, detEnabled };
}

function isSeriesSelected(meta, maps){
    const dsOk = maps.datasetEnabled[meta.datasetName || 'Dataset'] !== false;
    const clsOk = maps.classEnabled[meta.cls + '||' + (meta.datasetName || '')] !== false;
    const bandOk = maps.bandEnabled[meta.label + '||' + (meta.datasetName || '')] !== false;
    const detOk = (maps.detEnabled ? maps.detEnabled[meta.det + '||' + (meta.datasetName || '')] : true) !== false;
    return dsOk && clsOk && bandOk && detOk;
}

function buildControlPanel(chart, series){
    const panel = document.getElementById('control-panel'); if(!panel) return;
    const datasets = Array.from(new Set(series.map(s=>s.datasetName || 'Dataset').filter(Boolean)));
    // 读取频段分组配置：优先 emcData.chart_config.frequency_groups，否则用 DEFAULT_FREQUENCY_GROUPS
    const frequencyGroups = (emcData && emcData.chart_config && Array.isArray(emcData.chart_config.frequency_groups) && emcData.chart_config.frequency_groups.length > 0)
        ? emcData.chart_config.frequency_groups
        : (typeof DEFAULT_FREQUENCY_GROUPS !== 'undefined' ? DEFAULT_FREQUENCY_GROUPS : null);
    let html = '<div class="control-wrap">';
    html += '<div class="control-title">可勾选显示内容</div>';
    html += '<div class="control-actions"><button type="button" id="select-all">一键全选</button><button type="button" id="invert-selection">一键反选</button></div>';
    html += '<div class="dataset-grid">';

    datasets.forEach(function(dsName){
        const dsSeries = series.filter(s=> (s.datasetName || 'Dataset') === dsName);
        const dsIdx = dsSeries.length ? dsSeries[0].datasetIdx : 0;
        const dsColor = DATASET_PALETTE[dsIdx % DATASET_PALETTE.length];
        const classes = Array.from(new Set(dsSeries.map(s=>s.cls).filter(Boolean)));
        // 保留 band 元数据（bandLabel/bandRange + bandStart/bandEnd），用于按频率区间分组
        const bandMetaList = (() => {
            const seen = {}; const out = [];
            dsSeries.forEach(function(s){
                const label = s.bandLabel || s.bandRange || s.name;
                if (!label || seen[label]) return;
                seen[label] = true;
                out.push({ label: label, bandStart: Number(s.bandStart), bandEnd: Number(s.bandEnd) });
            });
            return out;
        })();
        const dets = Array.from(new Set(dsSeries.map(s=> s.det || '').filter(Boolean)));
        const dsId = 'ds_' + dsName.replace(/[^a-zA-Z0-9_-]/g, '_');
        html += '<div class="dataset-col" data-dataset="'+escapeHtml(dsName)+'">';
        html += '<div class="dataset-col-title"><span>'+escapeHtml(dsName)+'</span><span class="dataset-col-badge">'+dsSeries.length+' 条系列</span></div>';
        html += '<div class="dataset-col-section"><div class="control-heading">显示整组</div><div class="control-items">';
        html += '<label class="control-item"><input type="checkbox" id="'+dsId+'" checked data-kind="dataset" data-value="'+escapeHtml(dsName)+'"> '+escapeHtml(dsName)+'</label>';
        html += '</div></div>';
        if(classes.length > 0) {
            html += '<div class="dataset-col-section"><div class="control-heading">按 Class</div><div class="control-items">';
            classes.forEach(function(cls){ const id = dsId + '_cls_' + cls.replace(/[^a-zA-Z0-9_-]/g, '_'); const color = dsColor; html += '<label class="control-item"><input type="checkbox" id="'+id+'" checked data-kind="class" data-dataset="'+escapeHtml(dsName)+'" data-value="'+escapeHtml(cls)+'"><span class="class-dot" style="background:'+color+'"></span>'+escapeHtml(cls)+'</label>'; });
            html += '</div></div>';
        }
        if(dets.length > 0) {
            html += '<div class="dataset-col-section"><div class="control-heading">按 Detector</div><div class="control-items">';
            dets.forEach(function(dt){
                const id = dsId + '_det_' + dt;
                const lineStyle = LINE_STYLES[dt] || 'solid';
                let lineStyleStr = '';
                if (lineStyle === 'dashed') lineStyleStr = 'border-top: 2px dashed currentColor; height: 2px; background: none;';
                else if (lineStyle === 'dotted') lineStyleStr = 'border-top: 2px dotted currentColor; height: 2px; background: none;';
                else lineStyleStr = 'background-color: currentColor; height: 2px; border: none;';
                html += '<label class="control-item" style="color:'+dsColor+'"><input type="checkbox" id="'+id+'" checked data-kind="det" data-dataset="'+escapeHtml(dsName)+'" data-value="'+escapeHtml(dt)+'"><span class="det-line" style="'+lineStyleStr+'"></span>'+escapeHtml(dt)+'</label>';
            });
            html += '</div></div>';
        }
        // 按测试/频段：按 frequencyGroups 分组（默认 4 组：低/中/高/超高频），无配置时扁平展示
        if (bandMetaList.length > 0) {
            html += '<div class="dataset-col-section"><div class="control-heading">按测试/频段</div>';
            const renderBandCheckbox = function(b) {
                const id = dsId + '_band_' + b.label.replace(/[^a-zA-Z0-9_-]/g, '_');
                return '<label class="control-item"><input type="checkbox" id="'+id+'" checked data-kind="band" data-dataset="'+escapeHtml(dsName)+'" data-value="'+escapeHtml(b.label)+'"> '+escapeHtml(b.label)+'</label>';
            };
            if (Array.isArray(frequencyGroups) && frequencyGroups.length > 0) {
                // 频段唯一归属：按 group 顺序，第一个区间与该 band 有交集的 group 拥有该 band
                // 解决"一个 band 跨多个 group"导致重复显示的问题
                // 例：band 25-200 MHz 跨 low/mid，只展示在 low（按 group 顺序）
                const assignedBand = new Set();
                frequencyGroups.forEach(function(g, gIdx){
                    const inGroup = [];
                    bandMetaList.forEach(function(b){
                        if (assignedBand.has(b.label)) return; // 已被前面 group 占用，跳过
                        const s = b.bandStart, e = b.bandEnd;
                        if (!isFinite(s) || !isFinite(e) || s <= 0 || e <= 0) return;
                        if (g.min != null && e < Number(g.min)) return;
                        if (g.max != null && s > Number(g.max)) return;
                        inGroup.push(b);
                        assignedBand.add(b.label);
                    });
                    const groupKey = g.id || ('g' + gIdx);
                    const toggleId = dsId + '_group_' + groupKey;
                    // 即使组内为空也显示折叠框，让用户看到完整分组结构
                    html += '<details class="frequency-group" data-group-id="'+escapeHtml(groupKey)+'" open>';
                    html += '<summary class="frequency-group-summary">';
                    html += '<span class="frequency-group-name">'+escapeHtml(g.name || g.id || '组')
                        + ' <span class="frequency-group-summary-count">'+inGroup.length+'</span></span>';
                    // 一键全选 checkbox：仅组内有 band 时显示
                    if (inGroup.length > 0) {
                        html += '<label class="frequency-group-toggle"><input type="checkbox" class="group-toggle" id="'+toggleId+'" checked data-group-id="'+escapeHtml(groupKey)+'"> 全选</label>';
                    }
                    html += '</summary>';
                    if (inGroup.length > 0) {
                        html += '<div class="frequency-group-body">';
                        inGroup.forEach(function(b){ html += renderBandCheckbox(b); });
                        html += '</div>';
                    } else {
                        html += '<div class="frequency-group-body" style="color:#aaa;font-size:11px;">（无）</div>';
                    }
                    html += '</details>';
                });
                // 处理未匹配任何分组的 band（group 区间都不与 band 相交，如异常数据 / min/max 倒置）
                const unassignedBands = bandMetaList.filter(function(b){ return !assignedBand.has(b.label); });
                if (unassignedBands.length > 0) {
                    const toggleId = dsId + '_group_unassigned';
                    html += '<details class="frequency-group" data-group-id="unassigned" open>';
                    html += '<summary class="frequency-group-summary">';
                    html += '<span class="frequency-group-name">未分组 <span class="frequency-group-summary-count">'+unassignedBands.length+'</span></span>';
                    html += '<label class="frequency-group-toggle"><input type="checkbox" class="group-toggle" id="'+toggleId+'" checked data-group-id="unassigned"> 全选</label>';
                    html += '</summary>';
                    html += '<div class="frequency-group-body">';
                    unassignedBands.forEach(function(b){ html += renderBandCheckbox(b); });
                    html += '</div>';
                    html += '</details>';
                }
            } else {
                // 兜底：扁平渲染（与旧行为一致）
                html += '<div class="control-items">';
                bandMetaList.forEach(function(b){ html += renderBandCheckbox(b); });
                html += '</div>';
            }
            html += '</div>';
        }
    });

    html += '</div><div class="control-hint">取消勾选后，对应系列将隐藏；每个数据组独立分栏显示。</div>';
    html += '</div>';
    panel.innerHTML = html;

    const restoredFilters = restoreState && restoreState.typeA && restoreState.typeA.filters;
    restoreTypeAFilters(panel, restoredFilters);

    function applyFilters(shouldEmit){
        const maps = getSelectionMaps(panel);
        const newSeries = series.map(function(s, idx){
            const meta = getSeriesMeta(series, idx);
            const visible = isSeriesSelected(meta, maps);
            s.visible = visible;
            return Object.assign({}, s, { visible: visible, silent: !visible, tooltip: Object.assign({}, s.tooltip, { show: visible }), lineStyle: Object.assign({}, s.lineStyle, { opacity: visible?1:0 }), label: Object.assign({}, s.label, { show: visible }) });
        });
        chart.setOption({ series: newSeries }, false, true);
        if (window._typeAReadoutFrequency != null) renderTypeAReadout(chart, series, window._typeAReadoutFrequency);
        if (shouldEmit !== false) emitTypeAState(panel);
    }

    panel.applyFilters = applyFilters;
    // 一键全选 checkbox 修复：
    // 1) label 上 click 阻止冒泡到 summary（避免 <details> 折叠/展开）
    // 2) group-toggle change 用捕获阶段，确保 band 状态先于通用 applyFilters 更新
    panel.querySelectorAll('.frequency-group-toggle').forEach(function(labelEl){
        labelEl.addEventListener('click', function(e){ e.stopPropagation(); });
    });
    panel.querySelectorAll('input.group-toggle').forEach(function(toggleEl){
        toggleEl.addEventListener('change', function(){
            const groupDetails = toggleEl.closest('details.frequency-group');
            if (!groupDetails) return;
            const checked = toggleEl.checked;
            // 同步组内所有 band checkbox 的勾选状态
            groupDetails.querySelectorAll('input[type="checkbox"][data-kind="band"]').forEach(function(b){
                b.checked = checked;
            });
        }, true);
    });
    panel.querySelectorAll('input[type="checkbox"]').forEach(el=> el.addEventListener('change', function(){ applyFilters(true); }));

    const selectAllBtn = panel.querySelector('#select-all');
    if(selectAllBtn) selectAllBtn.addEventListener('click', function(){
        panel.querySelectorAll('input[type="checkbox"]').forEach(el=> el.checked = true);
        applyFilters(true);
    });

    const invertBtn = panel.querySelector('#invert-selection');
    if(invertBtn) invertBtn.addEventListener('click', function(){
        panel.querySelectorAll('input[type="checkbox"]').forEach(el=> el.checked = !el.checked);
        applyFilters(true);
    });

    if (restoredFilters) applyFilters(false);

    return panel;
}

function interpolateSeriesValue(series, frequency) {
    if (!series || series.visible === false || !isFinite(Number(frequency))) return null;
    const x = Number(frequency);
    if (x < Number(series.bandStart) || x > Number(series.bandEnd)) return null;
    const points = (series.data || []).filter(function(point){
        return Array.isArray(point) && point.length >= 2 && isFinite(Number(point[0])) && isFinite(Number(point[1]));
    }).map(function(point){ return [Number(point[0]), Number(point[1])]; }).sort(function(a,b){ return a[0]-b[0]; });
    if (!points.length || x < points[0][0] || x > points[points.length-1][0]) return null;
    for (let index=0; index<points.length; index+=1) {
        const current = points[index];
        if (Math.abs(current[0]-x) <= Math.max(1, Math.abs(x))*1e-12) return current[1];
        if (current[0] > x && index > 0) {
            const previous = points[index-1];
            if (current[0] === previous[0]) return previous[1];
            const ratio = (x-previous[0])/(current[0]-previous[0]);
            return previous[1] + ratio*(current[1]-previous[1]);
        }
    }
    return points[points.length-1][1];
}

function formatReadoutValue(value) {
    const number = Number(value);
    if (!isFinite(number)) return '';
    return number.toFixed(3).replace(/\\.?0+$/, '');
}

// 格式化 y 值（带单位）
function formatTooltipYValue(value, unit) {
    const formatted = formatReadoutValue(value);
    return unit ? (formatted + ' ' + unit) : formatted;
}

function typeAFrequencyFromPixel(chart, pixel) {
    const axisValue = chart.convertFromPixel({xAxisIndex:0}, pixel);
    const directFrequency = Array.isArray(axisValue) ? Number(axisValue[0]) : Number(axisValue);
    if (isFinite(directFrequency) && directFrequency > 0) return directFrequency;
    const gridValue = chart.convertFromPixel({gridIndex:0}, pixel);
    const gridFrequency = Array.isArray(gridValue) ? Number(gridValue[0]) : Number(gridValue);
    return isFinite(gridFrequency) && gridFrequency > 0 ? gridFrequency : null;
}

// 统一入口：设置 axisPointer 并渲染读数面板
function showReadout(chart, series, frequency) {
    try {
        chart.setOption({ xAxis:{ axisPointer:{ show:true, status:'show', value:frequency, label:{show:false} } } });
        renderTypeAReadout(chart, series, frequency);
        renderTypeABandwidth(series, frequency);
    } catch(e) { console.error('[showReadout] error:', e); }
}

function renderTypeAReadout(chart, series, frequency) {
    try {
    var panel = document.getElementById('type-a-readout');
    if (!panel) return;
    const rows = visibleTypeASeries(series).map(function(item){
        return { series:item, value:interpolateSeriesValue(item, frequency) };
    }).filter(function(row){ return row.value != null && isFinite(Number(row.value)); });
    rows.sort(function(a,b){
        return String(a.series.datasetName).localeCompare(String(b.series.datasetName)) || String(a.series.det).localeCompare(String(b.series.det)) || Number(a.series.bandStart)-Number(b.series.bandStart);
    });
    window._typeAReadoutFrequency = Number(frequency);
    let html = '<div class="type-a-readout-header"><div class="type-a-readout-title">点击位置：'+escapeHtml(formatFrequencyTick(frequency))+'<span class="type-a-readout-meta">'+rows.length+' 条有效曲线</span></div>'
        + '<div class="type-a-readout-freq">'
        + '<input id="type-a-freq-input" type="text" placeholder="如 100" />'
        + '<span class="unit">MHz</span>'
        + '<button id="type-a-freq-view">查看</button>'
        + '</div>'
        + '<button type="button" class="type-a-readout-close" aria-label="关闭读数">关闭</button></div>';
    html += '<div class="type-a-readout-table-wrap"><table class="type-a-readout-table"><thead><tr><th>曲线</th><th>标准来源</th><th>Detector / Class</th><th>频段</th><th>纵轴值</th></tr></thead><tbody>';
    rows.forEach(function(row){
        const item = row.series;
        const style = item.lineStyle || {};
        const lineType = style.type === 'dashed' ? 'dashed' : (style.type === 'dotted' ? 'dotted' : 'solid');
        html += '<tr><td><span class="type-a-readout-swatch" style="color:'+escapeHtml(style.color||'#333')+';border-top-style:'+lineType+'"></span></td><td>'+escapeHtml(item.datasetName)+'</td><td>'+escapeHtml((item.det||'-')+' / '+(item.cls||'-'))+'</td><td>'+escapeHtml(item.bandLabel||item.bandRange||'-')+'</td><td>'+escapeHtml(formatReadoutValue(row.value)+' '+(item.unit||''))+'</td></tr>';
    });
    html += '</tbody></table></div>';

    // 差值对比：独立面板 #type-a-diff
    var diffPanel = document.getElementById('type-a-diff');
    if (diffPanel) {
        var allSeriesRows = (Array.isArray(series) ? series : []).map(function(s){
            return { series: s, value: interpolateSeriesValue(s, frequency) };
        });
        var datasetNames = Array.from(new Set(allSeriesRows.map(function(r){ return r.series.datasetName; })));
        var diffHtml = '';
        if (datasetNames.length === 2) {
            var dsA = datasetNames[0];
            var dsB = datasetNames[1];
            // 提取各数据集的单位，检查是否兼容
            // 归一化单位：去括号、统一 µ (U+00B5) 和 μ (U+03BC) 为 µ (U+00B5)
            function normalizeUnit(raw) {
                return (raw || '').trim()
                    .replace(/[\(\)]/g, '')          // 去括号: dB(µV/m) → dBµV/m
                    .replace(/\u03BC/g, '\u00B5');   // 统一 Unicode µ
            }
            var unitByDs = {};
            allSeriesRows.forEach(function(row){
                var s = row.series;
                var u = normalizeUnit(s.unit);
                if (u) {
                    if (!unitByDs[s.datasetName]) unitByDs[s.datasetName] = [];
                    if (unitByDs[s.datasetName].indexOf(u) === -1) unitByDs[s.datasetName].push(u);
                }
            });
            var unitsA = unitByDs[dsA] || [];
            var unitsB = unitByDs[dsB] || [];
            var primaryUnitA = unitsA.length > 0 ? unitsA[0] : '';
            var primaryUnitB = unitsB.length > 0 ? unitsB[0] : '';
            var unitsCompatible = primaryUnitA && primaryUnitB && primaryUnitA === primaryUnitB;
            if (!unitsCompatible) {
                diffHtml += '<div class="type-a-diff-indicator">差值计算已禁用：单位不同 (' + escapeHtml(primaryUnitA || '未知') + ' vs ' + escapeHtml(primaryUnitB || '未知') + ')，无法进行差值对比</div>';
            } else {
            var grouped = {};
            // 辅助函数：判断 cls 是否为默认值（无分类）
            function isDefaultCls(c) { return !c || c === 'Default' || c === '__default__'; }
            function normalizeCls(c) { return isDefaultCls(c) ? '__default__' : c; }
            allSeriesRows.forEach(function(row){
                if (row.value == null || !isFinite(Number(row.value))) return;
                var s = row.series;
                var clsKey = normalizeCls(s.cls);
                var gKey = s.datasetName + '||' + s.det + '||' + clsKey;
                if (!grouped[gKey] || row.value < grouped[gKey].value) {
                    grouped[gKey] = row;
                }
            });
            var grpA = []; var grpB = [];
            Object.keys(grouped).forEach(function(k){
                var r = grouped[k];
                if (r.series.datasetName === dsA) grpA.push(r);
                else if (r.series.datasetName === dsB) grpB.push(r);
            });
            var diffs = [];
            grpA.forEach(function(rA){
                var detA = rA.series.det;
                var clsA = normalizeCls(rA.series.cls);
                var isDefaultA = isDefaultCls(rA.series.cls);
                grpB.forEach(function(rB){
                    var detB = rB.series.det;
                    var clsB = normalizeCls(rB.series.cls);
                    var isDefaultB = isDefaultCls(rB.series.cls);
                    if (detA !== detB) return;
                    if (!isDefaultA && !isDefaultB && clsA !== clsB) return;
                    diffs.push({
                        det: detA, clsA: isDefaultA ? 'default' : clsA, clsB: isDefaultB ? 'default' : clsB,
                        bandA: rA.series.bandLabel || rA.series.bandRange || '-',
                        bandB: rB.series.bandLabel || rB.series.bandRange || '-',
                        valA: rA.value, valB: rB.value,
                        diff: rA.value - rB.value,
                        unit: rA.series.unit || rB.series.unit || ''
                    });
                });
            });
            diffHtml += '<div class="type-a-diff-indicator">差值计算已执行 (' + escapeHtml(dsA) + ' vs ' + escapeHtml(dsB) + ')，分组: ' + grpA.length + ' + ' + grpB.length + ' 条，匹配到 ' + diffs.length + ' 组</div>';
            if (diffs.length > 0) {
                diffs.sort(function(a,b){ return a.det.localeCompare(b.det) || a.clsA.localeCompare(b.clsA) || a.clsB.localeCompare(b.clsB); });
                diffHtml += '<div class="type-a-diff-section"><div class="type-a-diff-title">差值对比 (' + escapeHtml(dsA) + ' - ' + escapeHtml(dsB) + ')</div>';
                diffHtml += '<div class="type-a-readout-table-wrap"><table class="type-a-readout-table type-a-diff-table"><thead><tr><th>Detector</th><th>' + escapeHtml(dsA) + ' Class</th><th>' + escapeHtml(dsA) + ' 曲线</th><th>' + escapeHtml(dsB) + ' Class</th><th>' + escapeHtml(dsB) + ' 曲线</th><th>' + escapeHtml(dsA) + ' 值</th><th>' + escapeHtml(dsB) + ' 值</th><th>差值</th></tr></thead><tbody>';
                diffs.forEach(function(d){
                    var diffSign = d.diff > 0 ? '+' : '';
                    var diffClass = d.diff > 0 ? 'type-a-diff-pos' : (d.diff < 0 ? 'type-a-diff-neg' : 'type-a-diff-zero');
                    var unit = d.unit;
                    diffHtml += '<tr><td>' + escapeHtml(d.det) + '</td><td>' + escapeHtml(d.clsA) + '</td><td>' + escapeHtml(d.bandA) + '</td><td>' + escapeHtml(d.clsB) + '</td><td>' + escapeHtml(d.bandB) + '</td><td>' + escapeHtml(formatReadoutValue(d.valA) + ' ' + unit) + '</td><td>' + escapeHtml(formatReadoutValue(d.valB) + ' ' + unit) + '</td><td class="' + diffClass + '">' + escapeHtml(diffSign + formatReadoutValue(d.diff) + ' ' + unit) + '</td></tr>';
                });
                diffHtml += '</tbody></table></div></div>';
            }
            } // end else (unitsCompatible)
        } else {
            diffHtml += '<div class="type-a-diff-indicator">差值计算未执行：需要恰好 2 个标准（当前 ' + datasetNames.length + ' 个）</div>';
        }
        diffPanel.innerHTML = diffHtml;
        diffPanel.hidden = false;
    }
    panel.innerHTML = html;
    panel.hidden = false;
    // 精确频率输入框事件绑定
    var freqInput = panel.querySelector('#type-a-freq-input');
    var freqViewBtn = panel.querySelector('#type-a-freq-view');
    if (freqInput && freqViewBtn) {
        function onFreqView() {
            var val = parseFloat(freqInput.value.trim());
            if (isNaN(val) || val <= 0) {
                freqInput.style.borderColor = '#e74c3c';
                return;
            }
            freqInput.style.borderColor = '';
            showReadout(chart, series, val);
        }
        freqViewBtn.onclick = onFreqView;
        freqInput.onkeydown = function(e) { if (e.key === 'Enter') onFreqView(); };
    }
    const closeButton = panel.querySelector('.type-a-readout-close');
    if (closeButton) closeButton.onclick = function(){
        panel.hidden = true;
        panel.innerHTML = '';
        window._typeAReadoutFrequency = null;
        chart.setOption({ xAxis:{ axisPointer:{ status:'hide' } } });
        var diffPanel = document.getElementById('type-a-diff');
        if (diffPanel) { diffPanel.hidden = true; diffPanel.innerHTML = ''; }
        // 同步关闭带宽对比面板
        var bwPanel = document.getElementById('type-a-bandwidth');
        if (bwPanel) { bwPanel.hidden = true; bwPanel.innerHTML = ''; }
    };
    } catch(e) { console.error('[renderTypeAReadout] error:', e); }
}

// 带宽对比面板：列出当前 x 位置所有可见曲线的 bw，并高亮带宽不一致行
// 决策说明：bw 字符串原始对比（"9 kHz" vs "120 kHz"），不去解析单位；差异判定 = 非"—"值 Set.size > 1
function renderTypeABandwidth(series, frequency) {
    try {
        var panel = document.getElementById('type-a-bandwidth');
        if (!panel) return;
        var freq = Number(frequency);
        if (!isFinite(freq) || freq <= 0) { panel.hidden = true; return; }

        // 收集当前 x 位置命中的可见 series（与 renderTypeAReadout 一致：bandStart/End 区间内）
        var rows = visibleTypeASeries(series).map(function(s){
            var f1 = Number(s.bandStart), f2 = Number(s.bandEnd);
            if (!isFinite(f1) || !isFinite(f2) || f1 <= 0 || f2 <= 0) return null;
            if (freq < f1 || freq > f2) return null;
            return {
                name: s.name || '-',
                dataset: s.datasetName || '-',
                det: s.det || '-',
                cls: s.cls || '-',
                band: s.bandLabel || s.bandRange || '-',
                bw: s.bw || '—'
            };
        }).filter(Boolean);

        if (rows.length === 0) { panel.hidden = true; panel.innerHTML = ''; return; }

        // 计算"非空带宽值"的众数（出现次数最多的 bw）
        var nonEmpty = rows.map(function(r){ return r.bw; }).filter(function(b){ return b && b !== '—'; });
        var dominant = null;
        if (nonEmpty.length > 0) {
            var counts = {};
            nonEmpty.forEach(function(b){ counts[b] = (counts[b] || 0) + 1; });
            var maxCount = -1;
            Object.keys(counts).forEach(function(k){ if (counts[k] > maxCount) { maxCount = counts[k]; dominant = k; } });
        }
        // 差异判定：去重后 > 1 种 bw（忽略"—"占位）即视为存在差异
        var distinctBws = Array.from(new Set(nonEmpty));
        var hasMismatch = distinctBws.length > 1;

        // 渲染标题栏（异常用红色警告，一致用绿色提示）
        var titleBadge;
        if (rows.every(function(r){ return r.bw === '—'; })) {
            titleBadge = '<span class="type-a-bandwidth-warn">⚠ 当前频段无带宽数据</span>';
        } else if (hasMismatch) {
            titleBadge = '<span class="type-a-bandwidth-warn">⚠ 检测到不同带宽（' + escapeHtml(distinctBws.join(' / ')) + '）</span>';
        } else {
            titleBadge = '<span class="type-a-bandwidth-ok">✓ 带宽一致：' + escapeHtml(dominant || '—') + '</span>';
        }

        var html = '<div class="type-a-readout-header">'
            + '<div class="type-a-readout-title">带宽对比（当前频率 ' + escapeHtml(formatFrequencyTick(freq))
            + '）<span class="type-a-readout-meta">' + rows.length + ' 条曲线</span></div>'
            + titleBadge
            + '</div>';

        html += '<div class="type-a-readout-table-wrap"><table class="type-a-readout-table"><thead><tr>'
            + '<th>曲线</th><th>标准来源</th><th>Detector/Class</th><th>频段</th><th>带宽</th></tr></thead><tbody>';
        rows.forEach(function(r){
            var isMismatch = hasMismatch && r.bw !== '—' && r.bw !== dominant;
            var rowCls = isMismatch ? ' class="type-a-bandwidth-row-mismatch"' : '';
            html += '<tr' + rowCls + '>'
                + '<td>' + escapeHtml(r.name) + '</td>'
                + '<td>' + escapeHtml(r.dataset) + '</td>'
                + '<td>' + escapeHtml(r.det + ' / ' + r.cls) + '</td>'
                + '<td>' + escapeHtml(r.band) + '</td>'
                + '<td>' + escapeHtml(r.bw) + '</td>'
                + '</tr>';
        });
        html += '</tbody></table></div>';

        panel.innerHTML = html;
        panel.hidden = false;
    } catch(e) { console.error('[renderTypeABandwidth] error:', e); }
}

function renderTypeA(typeADatasets){
    const series = buildAllSeries({ datasets: typeADatasets });
    if (series.length === 0) {
        document.getElementById('chart').textContent = '没有生成 Type A 数据系列，请检查输入数据。';
        return;
    }
    const chart = echarts.init(document.getElementById('chart'));
    window._chart = chart;
    window._series = series;

    const datasetNames = Array.from(new Set(series.map(s => s.datasetName || 'Dataset')));

    let yMinR, yMaxR, yUnit = 'dBµV/m';
    const cfg = (emcData && emcData.chart_config) || null;
    if (cfg && typeof cfg.y_min === 'number' && typeof cfg.y_max === 'number') {
        yMinR = cfg.y_min;
        yMaxR = cfg.y_max;
        if (cfg.y_unit) yUnit = cfg.y_unit;
    } else {
        let yMin = Infinity, yMax = -Infinity;
        series.forEach(function(s){
            (s.data || []).forEach(function(p){
                const y = p[1];
                if (typeof y === 'number' && isFinite(y)) {
                    if (y < yMin) yMin = y;
                    if (y > yMax) yMax = y;
                }
            });
        });
        if (!isFinite(yMin) || !isFinite(yMax)) { yMinR = 0; yMaxR = 120; }
        else if (yMin === yMax) { yMinR = Math.floor(yMin - 5); yMaxR = Math.ceil(yMax + 5); }
        else {
            const span = yMax - yMin;
            let pad = Math.max(5, span * 0.1);
            if (yMin < 0 && yMin > -pad) pad = Math.max(pad, Math.abs(yMin) + 3);
            yMinR = Math.floor((yMin - pad) / 5) * 5;
            yMaxR = Math.ceil((yMax + pad) / 5) * 5;
        }
    }

    const xUnion = cfg && Array.isArray(cfg.x_union) ? cfg.x_union : typeARangeFromSeries(series);
    const xCommon = cfg && Array.isArray(cfg.x_intersection) ? cfg.x_intersection : null;
    const restoredTypeA = restoreState && restoreState.typeA;
    let initialRange = xUnion ? xUnion.slice() : null;
    if (restoredTypeA && xUnion) {
        const restoredStart = Number(restoredTypeA.startValue);
        const restoredEnd = Number(restoredTypeA.endValue);
        const clampedStart = Math.max(Number(xUnion[0]), restoredStart);
        const clampedEnd = Math.min(Number(xUnion[1]), restoredEnd);
        if (isFinite(clampedStart) && isFinite(clampedEnd) && clampedStart > 0 && clampedEnd >= clampedStart) {
            initialRange = [clampedStart, clampedEnd];
        }
    }
    window._typeARange = initialRange ? initialRange.slice() : null;
    const option = {
        // 禁用 ECharts 内置 tooltip，改用下方自定义 HTML 浮层
        // 原因：trigger:'item' 在密集多线场景下几乎无法精准命中；showTip action 与 displayMode:'single' 兼容性差
        tooltip: { show: false },
        grid: { left:'6%', right:'4%', top:68, bottom:64, containLabel:true },
        xAxis: { type: 'log', logBase:10, name: '频率', nameLocation:'middle', nameGap:38, min:xUnion ? xUnion[0] : null, max:xUnion ? xUnion[1] : null, axisLabel: { formatter: formatFrequencyTick } },
        yAxis: { type: 'value', name: yUnit, nameLocation:'middle', nameGap:50, min: yMinR, max: yMaxR, splitNumber: 8 },
        dataZoom: [
            { id:'type-a-inside', type:'inside', xAxisIndex:[0], startValue:initialRange ? initialRange[0] : null, endValue:initialRange ? initialRange[1] : null, zoomOnMouseWheel:true, moveOnMouseMove:true, minValueSpan: xUnion ? (xUnion[0] * 0.5) : 0.1, maxValueSpan: xUnion ? (xUnion[1] - xUnion[0]) : null },
            { id:'type-a-slider', type:'slider', xAxisIndex:[0], startValue:initialRange ? initialRange[0] : null, endValue:initialRange ? initialRange[1] : null, height:20, bottom:10, borderColor:'#ddd', backgroundColor:'#fafafa', fillerColor:'rgba(200,230,255,0.4)', handleStyle:{ color:'#1677ff' }, minValueSpan: xUnion ? (xUnion[0] * 0.5) : 0.1, maxValueSpan: xUnion ? (xUnion[1] - xUnion[0]) : null }
        ],
        series: series
    };

    chart.setOption(option);

    // 图片叠加（如果配置了 overlayConfig）
    if (overlayConfig && overlayConfig.imageDataUrl) {
        try {
            var xMinPx = chart.convertToPixel({ xAxisIndex: 0 }, overlayConfig.axis.xMin);
            var xMaxPx = chart.convertToPixel({ xAxisIndex: 0 }, overlayConfig.axis.xMax);
            var yMinPx = chart.convertToPixel({ yAxisIndex: 0 }, overlayConfig.axis.yMax);
            var yMaxPx = chart.convertToPixel({ yAxisIndex: 0 }, overlayConfig.axis.yMin);
            chart.setOption({
                graphic: [{
                    type: 'image',
                    id: 'overlay-image',
                    left: xMinPx,
                    top: yMinPx,
                    style: {
                        image: overlayConfig.imageDataUrl,
                        width: xMaxPx - xMinPx,
                        height: yMaxPx - yMinPx,
                        opacity: overlayConfig.opacity
                    },
                    z: 100
                }]
            });
        } catch (e) {
            console.warn('图片叠加失败:', e.message);
        }
    }

    // hover 增强：监听 zrender 鼠标移动，自动找距离鼠标 y 最近的可见 series
    // 1) ECharts 内部 emphasis.focus:'series' 自动淡化其他曲线并加粗当前曲线
    // 2) 自定义 HTML 浮层显示该曲线的名称/类型/值（ECharts 5 showTip action 与 displayMode:'single' 兼容性差）
    (function() {
        try {
            var zr = chart.getZr();
            var chartDom = chart.getDom();
            // 1) 创建浮层 DOM
            var tip = document.createElement('div');
            tip.className = 'type-a-hover-tip';
            tip.style.cssText = 'position:absolute;z-index:1000;pointer-events:none;display:none;'
                + 'background:rgba(50,50,50,0.94);color:#fff;padding:8px 12px;border-radius:6px;'
                + 'box-shadow:0 4px 16px rgba(0,0,0,0.25);font-size:12px;line-height:1.6;'
                + 'border:1px solid #333;max-width:360px;';
            chartDom.parentNode.appendChild(tip);

            // 2) 更新浮层内容
            function renderTipContent(s, x, y) {
                var color = (s.lineStyle && s.lineStyle.color) || '#333';
                var yValue = interpolateSeriesValue(s, x);
                var yText = yValue != null ? (formatReadoutValue(yValue) + (s.unit ? ' ' + s.unit : '')) : '-';
                var detLabel = s.det || 'Unknown';
                var clsLabel = s.cls || 'Default';
                var typeText = clsLabel && clsLabel !== 'Default' ? (detLabel + ' / ' + clsLabel) : detLabel;
                var nameText = s.bandLabel || s.bandRange || s.name || '-';
                var freqText = formatFrequencyTick(x);
                tip.innerHTML = [
                    '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">',
                    '  <span style="display:inline-block;width:20px;height:0;border-top:3px solid ' + color + ';"></span>',
                    '  <span style="font-weight:600;font-size:13px;">' + escapeHtml(s.datasetName || '') + '</span>',
                    '</div>',
                    '<div style="line-height:1.7;">',
                    '  <div><span style="color:#bbb;">名称：</span>' + escapeHtml(nameText) + '</div>',
                    '  <div><span style="color:#bbb;">类型：</span>' + escapeHtml(typeText) + '</div>',
                    '  <div><span style="color:#bbb;">频率：</span>' + escapeHtml(freqText) + '</div>',
                    '  <div><span style="color:#bbb;">带宽：</span>' + escapeHtml(s.bw || '—') + '</div>',
                    '  <div><span style="color:#bbb;">值：</span><span style="color:#fff;font-weight:600;">' + yText + '</span></div>',
                    '</div>'
                ].join('');
            }

            // 3) 鼠标移动：找最近 series + 更新浮层位置和内容
            var lastHighlightedIdx = -1;
            // 从 grid 范围 + dataZoom 范围手动反算坐标（ECharts convertFromPixel 在某些情况下返回 null，用此兜底）
            function pixelToValue(pixelX, pixelY) {
                var opt = chart.getOption();
                var xAxisOpt = (opt.xAxis && opt.xAxis[0]) || {};
                var yAxisOpt = (opt.yAxis && opt.yAxis[0]) || {};
                var dz = (opt.dataZoom && opt.dataZoom[0]) || {};
                // 当前显示 x 范围（考虑 dataZoom）
                var dataStart = Number(dz.startValue);
                var dataEnd = Number(dz.endValue);
                if (!isFinite(dataStart) || !isFinite(dataEnd) || dataStart <= 0 || dataEnd <= dataStart) {
                    dataStart = Number(xAxisOpt.min) || 0.001;
                    dataEnd = Number(xAxisOpt.max) || 1;
                }
                // 当前显示 y 范围
                var yMin = Number(yAxisOpt.min);
                var yMax = Number(yAxisOpt.max);
                if (!isFinite(yMin) || !isFinite(yMax) || yMax <= yMin) { yMin = 0; yMax = 100; }
                // 从 chart 的 _view 区域拿 grid bounds（ECharts 私有但稳定）
                var gridInfo = chart._view && chart._view.series && chart._view.series[0] && chart._view.series[0].__view;
                // 直接用 convertToPixel 反推：找一个已知数据点求 grid bounds
                var testPt = [dataStart, (yMin + yMax) / 2];
                var testPx = chart.convertToPixel({xAxisIndex: 0, yAxisIndex: 0}, testPt);
                var xPixelStart, xPixelEnd, yPixelStart, yPixelEnd;
                if (testPx && isFinite(testPx[0])) {
                    xPixelStart = testPx[0];
                }
                testPt = [dataEnd, (yMin + yMax) / 2];
                testPx = chart.convertToPixel({xAxisIndex: 0, yAxisIndex: 0}, testPt);
                if (testPx && isFinite(testPx[0])) {
                    xPixelEnd = testPx[0];
                }
                // 用 yAxis 的极值反算 y 像素范围
                testPt = [(dataStart + dataEnd) / 2, yMax];
                testPx = chart.convertToPixel({xAxisIndex: 0, yAxisIndex: 0}, testPt);
                if (testPx && isFinite(testPx[1])) yPixelStart = testPx[1];
                testPt = [(dataStart + dataEnd) / 2, yMin];
                testPx = chart.convertToPixel({xAxisIndex: 0, yAxisIndex: 0}, testPt);
                if (testPx && isFinite(testPx[1])) yPixelEnd = testPx[1];
                if (!isFinite(xPixelStart) || !isFinite(xPixelEnd) || xPixelEnd === xPixelStart) return null;
                if (!isFinite(yPixelStart) || !isFinite(yPixelEnd) || yPixelStart === yPixelEnd) return null;
                // log scale 反算 x
                var ratio = (pixelX - xPixelStart) / (xPixelEnd - xPixelStart);
                var x = Math.pow(10, Math.log10(dataStart) + ratio * (Math.log10(dataEnd) - Math.log10(dataStart)));
                // y 是 value scale（不一定 log）
                var yRatio = (pixelY - yPixelStart) / (yPixelEnd - yPixelStart);
                var y = yMax + yRatio * (yMin - yMax);
                return { x: x, y: y };
            }
            function handleMove(pixelX, pixelY) {
                var coord = pixelToValue(pixelX, pixelY);
                if (!coord) { tip.style.display = 'none'; return; }
                var x = coord.x; var y = coord.y;
                if (!isFinite(x) || !isFinite(y) || x <= 0) { tip.style.display = 'none'; return; }
                // 找 y 值最接近鼠标位置的可见 series
                var bestIdx = -1; var bestDist = Infinity;
                for (var i = 0; i < series.length; i += 1) {
                    var s = series[i];
                    if (s.visible === false) continue;
                    var v = interpolateSeriesValue(s, x);
                    if (v == null || !isFinite(Number(v))) continue;
                    var d = Math.abs(Number(v) - y);
                    if (d < bestDist) { bestDist = d; bestIdx = i; }
                }
                if (bestIdx < 0) { tip.style.display = 'none'; return; }
                // 高亮当前 series（让 ECharts 自己做加粗+淡化）
                if (bestIdx !== lastHighlightedIdx) {
                    if (lastHighlightedIdx >= 0) chart.dispatchAction({ type: 'downplay', seriesIndex: lastHighlightedIdx });
                    chart.dispatchAction({ type: 'highlight', seriesIndex: bestIdx });
                    lastHighlightedIdx = bestIdx;
                }
                // 渲染浮层
                renderTipContent(series[bestIdx], x, y);
                // 定位浮层（跟随鼠标，偏移避免遮挡）
                var containerRect = chartDom.parentNode.getBoundingClientRect();
                var tipW = tip.offsetWidth || 240; var tipH = tip.offsetHeight || 120;
                var leftPx = pixelX + 14;
                var topPx = pixelY + 14;
                if (leftPx + tipW > containerRect.width) leftPx = pixelX - tipW - 14;
                if (topPx + tipH > containerRect.height) topPx = pixelY - tipH - 14;
                tip.style.left = Math.max(0, leftPx) + 'px';
                tip.style.top = Math.max(0, topPx) + 'px';
                tip.style.display = 'block';
            }
            function hideTip() {
                tip.style.display = 'none';
                if (lastHighlightedIdx >= 0) {
                    chart.dispatchAction({ type: 'downplay', seriesIndex: lastHighlightedIdx });
                    lastHighlightedIdx = -1;
                }
            }

            zr.on('mousemove', function(event) { handleMove(event.offsetX, event.offsetY); });
            zr.on('globalout', hideTip);
            // 兜底：dataZoom 缩放后像素映射失效，浮层先隐藏
            chart.on('datazoom', function() { hideTip(); });
            chart.on('restore', function() { hideTip(); });
            // 暴露 handleMove 到 chart 实例，便于外部测试和真实鼠标事件驱动
            try { chart._hoverTipHandle = handleMove; } catch(_e) {}
        } catch(e) { console.error('[hover-tip] init failed:', e); }
    })();

    // 缩放工具栏按钮
    var zoomToolbar = document.getElementById('type-a-zoom-toolbar');
    if (zoomToolbar) {
        zoomToolbar.innerHTML = '';
        var zoomInBtn = document.createElement('button');
        zoomInBtn.textContent = '放大';
        zoomInBtn.className = 'btn btn-default';
        zoomInBtn.title = '放大显示区域';
        zoomInBtn.addEventListener('click', function() {
            var range = window._typeARange || xUnion;
            var mid = (range[0] + range[1]) / 2;
            var span = (range[1] - range[0]) * 0.4;
            chart.dispatchAction({ type: 'dataZoom',
                startValue: Math.max(xUnion[0], mid - span / 2),
                endValue: Math.min(xUnion[1], mid + span / 2) });
        });
        zoomToolbar.appendChild(zoomInBtn);

        var zoomOutBtn = document.createElement('button');
        zoomOutBtn.textContent = '缩小';
        zoomOutBtn.className = 'btn btn-default';
        zoomOutBtn.title = '缩小显示区域';
        zoomOutBtn.addEventListener('click', function() {
            var range = window._typeARange || xUnion;
            var mid = (range[0] + range[1]) / 2;
            var span = (range[1] - range[0]) * 2.5;
            chart.dispatchAction({ type: 'dataZoom',
                startValue: Math.max(xUnion[0], mid - span / 2),
                endValue: Math.min(xUnion[1], mid + span / 2) });
        });
        zoomToolbar.appendChild(zoomOutBtn);

        var resetBtn = document.createElement('button');
        resetBtn.textContent = '重置';
        resetBtn.className = 'btn btn-default';
        resetBtn.title = '重置到全量范围';
        resetBtn.addEventListener('click', function() {
            applyTypeARange(chart, xUnion);
        });
        zoomToolbar.appendChild(resetBtn);
    }

    const controlPanel = buildControlPanel(chart, series);
    chart.on('datazoom', function(){
        const zoom = (chart.getOption().dataZoom || [])[0] || {};
        const startValue = Number(zoom.startValue);
        const endValue = Number(zoom.endValue);
        if (isFinite(startValue) && isFinite(endValue) && startValue > 0 && endValue >= startValue) {
            window._typeARange = [startValue, endValue];
            emitTypeAState(controlPanel);
        }
    });

    const chartDom = document.getElementById('chart');
    if (chartDom) {
        var dblClickFull = false;
        // 使用 addEventListener 而非 ondblclick，避免干扰单次 click 事件
        chartDom.addEventListener('dblclick', function() {
            if (dblClickFull) {
                applyTypeARange(chart, xUnion);
                dblClickFull = false;
            } else {
                var visibleRange = typeARangeFromSeries(series);
                if (visibleRange) applyTypeARange(chart, visibleRange);
                else applyTypeARange(chart, xUnion);
                dblClickFull = true;
            }
            setTimeout(function() { dblClickFull = false; }, 1500);
        });
    }
    // 点击图表显示读数面板
    if (chartDom) {
        // 捕获阶段，在 ECharts 内部处理之前触发
        chartDom.addEventListener('mousedown', function(event) {
            if (event.button !== 0) return;
            try {
                var frequency = typeAFrequencyFromPixel(chart, [event.offsetX, event.offsetY]);
                if (frequency !== null && frequency !== undefined) {
                    showReadout(chart, series, frequency);
                }
            } catch(e) { console.error('[readout mousedown]', e); }
        }, true);
        // zrender mousedown 备用（直接监听 ECharts 画布层）
        try {
            chart.getZr().on('mousedown', function(event) {
                var frequency = typeAFrequencyFromPixel(chart, [event.offsetX, event.offsetY]);
                if (frequency !== null && frequency !== undefined) {
                    showReadout(chart, series, frequency);
                }
            });
        } catch(e) { console.error('[readout zr]', e); }
    }

    window.addEventListener('resize', function(){ chart.resize(); });
}

function chooseTypeBAxisMode(mode, minFrequency, maxFrequency) {
    if (mode === 'log' || mode === 'value') return mode;
    return minFrequency > 0 && maxFrequency / minFrequency >= 100 ? 'log' : 'value';
}

// 状态颜色映射：A=绿(达标), B=黄(降级), C=红(失效)
var STATUS_COLORS = { 'A': '#27ae60', 'B': '#f39c12', 'C': '#e74c3c' };
function statusColor(status) { return STATUS_COLORS[String(status).toUpperCase()] || '#999'; }

// 按标准分组 methods，返回 { datasetName: [methods] }
function groupMethodsByDataset(methods) {
    var groups = {};
    methods.forEach(function(m) {
        var key = m.datasetName || 'Unknown';
        if (!groups[key]) groups[key] = [];
        groups[key].push(m);
    });
    return groups;
}

// 获取 BCI 方法区域（DBCI / CBCI 频率范围），用于 markArea 背景色带
function getBCIMethodZones(methods) {
    var dbciMin = Infinity, dbciMax = -Infinity;
    var cbciMin = Infinity, cbciMax = -Infinity;
    methods.forEach(function(m) {
        var meth = (m.method || '').toLowerCase();
        if (meth.indexOf('dbci') >= 0) {
            if (m.fmin < dbciMin) dbciMin = m.fmin;
            if (m.fmax > dbciMax) dbciMax = m.fmax;
        }
        if (meth.indexOf('cbci') >= 0) {
            if (m.fmin < cbciMin) cbciMin = m.fmin;
            if (m.fmax > cbciMax) cbciMax = m.fmax;
        }
    });
    var zones = [];
    if (isFinite(dbciMin)) zones.push({ name: 'DBCI', xMin: dbciMin, xMax: dbciMax, color: 'rgba(52,152,219,0.08)' });
    if (isFinite(cbciMin)) zones.push({ name: 'CBCI', xMin: cbciMin, xMax: cbciMax, color: 'rgba(243,156,18,0.08)' });
    return zones;
}

function renderTypeBFacets(typeBDatasets) {
    const container = document.getElementById('type-b-panel');
    if (!container) return;
    function facetStatusClass(status) { return status == null || status === '' ? 'status-null' : 'status-' + String(status).toLowerCase(); }
    function facetStatusText(status) { return status == null || status === '' ? '-' : String(status); }
    const facets = {};
    typeBDatasets.forEach(function(ds, datasetIdx){
        const methods = (ds.data && ds.data.test_methods) || [];
        methods.forEach(function(method){
            const family = method.method_family || 'needs_review';
            const quantity = method.physical_quantity || 'unknown';
            const unit = method.canonical_unit || method.unit || '';
            const facetKey = family + '||' + quantity + '||' + unit;
            if (!facets[facetKey]) facets[facetKey] = { key:facetKey, family:family, quantity:quantity, unit:unit, methods:[] };
            facets[facetKey].methods.push(Object.assign({}, method, { datasetName:ds.name, datasetIdx:datasetIdx }));
        });
    });
    const facetList = Object.keys(facets).sort().map(function(key){ return facets[key]; });
    // 标题：根据 family 显示中文
    var familyLabels = { free_field: '辐射抗扰度 (ALSE法)', bci: '大电流注入 (BCI)', magnetic_field: '磁场抗扰度', conducted_immunity: '传导抗扰度', needs_review: '待确认项目' };
    let html = '<div class="type-b-view"><h2>Type B 抗扰度对比</h2><div class="type-b-facet-grid">';
    facetList.forEach(function(facet, facetIdx){
        const chartId = 'type-b-facet-chart-' + facetIdx;
        const selectId = 'type-b-axis-mode-' + facetIdx;
        var displayTitle = familyLabels[facet.family] || facet.family;
        html += '<section class="type-b-facet" data-facet-key="'+escapeHtml(facet.key)+'">';
        html += '<div class="type-b-facet-header"><div><div class="type-b-facet-title">'+escapeHtml(displayTitle)+'</div>';
        html += '<div class="type-b-facet-meta">'+escapeHtml(facet.quantity)+' · '+escapeHtml(facet.unit || '单位待确认')+' · '+facet.methods.length+' 条要求</div></div>';
        html += '<label>横轴 <select class="type-b-axis-mode" id="'+selectId+'"><option value="auto">自动</option><option value="log">对数</option><option value="value">线性</option></select></label></div>';
        if (facet.family === 'needs_review' || facet.quantity === 'unknown') html += '<div class="type-b-review-note">方法或物理量尚未规范化，已隔离显示，不参与严格度直接比较。</div>';
        html += '<div class="type-b-chart" id="'+chartId+'"></div>';
        // 对于 free_field，增加严酷度对比列
        var isFreeField = facet.family === 'free_field';
        html += '<div class="type-b-table-wrapper"><table class="type-b-table"><thead><tr><th>来源</th><th>频段</th><th>等级</th><th>方法</th><th>调制</th><th>R1</th><th>R2</th><th>R3</th>' + (isFreeField ? '<th>严酷度</th>' : '') + '</tr></thead><tbody>';
        facet.methods.forEach(function(method){
            const status = method.function_status || {};
            html += '<tr><td>'+escapeHtml(method.datasetName)+'</td><td>'+escapeHtml(method.frequency)+'</td><td>'+escapeHtml(method.test_level)+'</td><td>'+escapeHtml(displayValue(method.method))+'</td><td>'+escapeHtml(displayValue(method.mod))+'</td>';
            ['R1','R2','R3'].forEach(function(region){ html += '<td class="'+facetStatusClass(status[region])+'">'+facetStatusText(status[region])+'</td>'; });
            if (isFreeField) {
                // 严酷度：值越小越严（限值越低要求越严格）
                var allVals = facet.methods.map(function(m){ return Number(m.value); }).filter(isFinite);
                var minVal = allVals.length ? Math.min.apply(null, allVals) : null;
                var severity = '';
                if (method.value != null && minVal != null) {
                    severity = Number(method.value) === minVal ? '更严' : '更松';
                }
                html += '<td class="' + (severity === '更严' ? 'status-c' : (severity === '更松' ? 'status-a' : '')) + '">' + severity + '</td>';
            }
            html += '</tr>';
        });
        html += '</tbody></table></div></section>';
    });
    html += '</div></div>';
    container.innerHTML = html;
    window._typeBCharts = [];

    facetList.forEach(function(facet, facetIdx){
        const chartDom = document.getElementById('type-b-facet-chart-' + facetIdx);
        const modeSelect = document.getElementById('type-b-axis-mode-' + facetIdx);
        if (!chartDom || !modeSelect) return;
        const positiveStarts = facet.methods.map(function(m){ return Number(m.fmin); }).filter(function(v){ return isFinite(v) && v > 0; });
        const positiveEnds = facet.methods.map(function(m){ return Number(m.fmax); }).filter(function(v){ return isFinite(v) && v > 0; });
        if (!positiveStarts.length || !positiveEnds.length) {
            chartDom.textContent = '没有可绘制的正频率范围；非正频率数据已保留在参数表中。';
            return;
        }
        const xMin = Math.min.apply(null, positiveStarts); const xMax = Math.max.apply(null, positiveEnds);
        const numericValues = facet.methods.map(function(m){ return Number(m.value); }).filter(isFinite);
        const yMinRaw = numericValues.length ? Math.min.apply(null, numericValues) : 0;
        const yMaxRaw = numericValues.length ? Math.max.apply(null, numericValues) : 100;
        const yPad = Math.max(5, (yMaxRaw-yMinRaw)*0.1);
        const isFreeField = facet.family === 'free_field';
        const isBCI = facet.family === 'bci';
        const isMagnetic = facet.family === 'magnetic_field';

        var series = [];
        if (isFreeField) {
            // ===== 辐射抗扰：分组柱状图（数值轴，按频率中点定位）=====
            var groups = groupMethodsByDataset(facet.methods);
            var dsNames = Object.keys(groups);
            dsNames.forEach(function(dsName, dsIdx) {
                var methods = groups[dsName];
                var barData = methods.map(function(m) {
                    var xMid = (Number(m.fmin) + Number(m.fmax)) / 2; // 频段中点
                    var yVal = m.value != null ? Number(m.value) : null;
                    var bestStatus = '';
                    var fs = m.function_status || {};
                    // 取最差的功能等级作为柱顶标注
                    ['R1','R2','R3'].forEach(function(r) {
                        var s = String(fs[r] || '').toUpperCase();
                        if (s === 'C' || (s === 'B' && bestStatus !== 'C') || (s === 'A' && !bestStatus)) bestStatus = s;
                    });
                    return { value: [xMid, yVal], fmin: m.fmin, fmax: m.fmax, level: m.test_level, status: bestStatus, method: m.method, mod: m.mod };
                }).filter(function(d) { return d.value[1] != null; });
                barData.sort(function(a, b) { return a.value[0] - b.value[0]; });
                var color = DATASET_PALETTE[dsIdx % DATASET_PALETTE.length];
                series.push({
                    name: dsName, type: 'bar', barWidth: '40%', barGap: '20%',
                    data: barData.map(function(d) { return d.value; }),
                    itemStyle: { color: color, borderRadius: [4, 4, 0, 0] },
                    label: {
                        show: true, position: 'top', fontSize: 11, fontWeight: 'bold',
                        formatter: function(p) {
                            var d = barData[p.dataIndex];
                            return d && d.status ? d.status : '';
                        },
                        color: function(p) {
                            var d = barData[p.dataIndex];
                            return d ? statusColor(d.status) : '#999';
                        }
                    },
                    tooltip: {
                        formatter: function(p) {
                            var d = barData[p.dataIndex];
                            if (!d) return '';
                            return '<b>' + dsName + '</b><br/>频段: ' + formatFrequencyTick(d.fmin) + ' ~ ' + formatFrequencyTick(d.fmax)
                                + '<br/>等级: ' + escapeHtml(d.level) + '<br/>场强: ' + d.value[1] + ' ' + (facet.unit || '')
                                + '<br/>功能等级: ' + (d.status || '-') + '<br/>方法: ' + escapeHtml(displayValue(d.method))
                                + '<br/>调制: ' + escapeHtml(displayValue(d.mod));
                        }
                    }
                });
            });
        } else {
            // ===== BCI / 磁场抗扰 / 其他：折线图（BCI加色带，磁场加面积填充）=====
            series = facet.methods.map(function(method, methodIdx){
                let points = Array.isArray(method.points) ? method.points.slice() : [];
                if (!points.length && method.value != null && method.fmin > 0 && method.fmax > 0) points = [[method.fmin, Number(method.value)], [method.fmax, Number(method.value)]];
                var s = { name:method.datasetName+' / '+method.test_level+' / '+method.frequency, type:'line', data:points, symbol:'none', connectNulls:false, lineStyle:{ color:DATASET_PALETTE[method.datasetIdx % DATASET_PALETTE.length], type:methodIdx % 2 ? 'dashed' : 'solid', width:2.2 } };
                // 磁场抗扰：添加面积填充
                if (isMagnetic) {
                    s.areaStyle = { color: DATASET_PALETTE[method.datasetIdx % DATASET_PALETTE.length], opacity: 0.25 };
                }
                return s;
            }).filter(function(item){ return item.data.length; });
        }

        const chart = echarts.init(chartDom);
        const restoredFacet = restoreState && restoreState.typeB && restoreState.typeB.facets && restoreState.typeB.facets[facet.key];
        const restoredMode = restoredFacet && ['auto','log','value'].indexOf(restoredFacet.mode) >= 0 ? restoredFacet.mode : 'auto';
        const restoredStart = restoredFacet ? Math.max(xMin, Number(restoredFacet.startValue)) : xMin;
        const restoredEnd = restoredFacet ? Math.min(xMax, Number(restoredFacet.endValue)) : xMax;
        const hasRestoredWindow = isFinite(restoredStart) && isFinite(restoredEnd) && restoredStart > 0 && restoredEnd >= restoredStart;
        const state = { mode:restoredMode, startValue:hasRestoredWindow ? restoredStart : xMin, endValue:hasRestoredWindow ? restoredEnd : xMax };
        modeSelect.value = state.mode;
        function emitFacetState(){
            emitChartState({ type:'typeB', facetKey:facet.key, mode:state.mode, startValue:state.startValue, endValue:state.endValue });
        }
        function applyAxisMode(){
            const axisType = chooseTypeBAxisMode(state.mode, xMin, xMax);
            var option = {
                tooltip:{ trigger:'axis', axisPointer:{type:'cross'} },
                grid:{ left:'7%', right:'4%', top:20, bottom:64, containLabel:true },
                xAxis:{ type:axisType, logBase:10, min:xMin, max:xMax, name:'频率', nameLocation:'middle', nameGap:38, axisLabel:{formatter:formatFrequencyTick} },
                yAxis:{ type:'value', name:facet.unit || '测试等级', nameLocation:'middle', nameGap:52, min:Math.floor((yMinRaw-yPad)/5)*5, max:Math.ceil((yMaxRaw+yPad)/5)*5 },
                dataZoom:[
                    { id:'type-b-inside-'+facetIdx, type:'inside', xAxisIndex:[0], startValue:state.startValue, endValue:state.endValue, zoomOnMouseWheel:true, moveOnMouseMove:true },
                    { id:'type-b-slider-'+facetIdx, type:'slider', xAxisIndex:[0], startValue:state.startValue, endValue:state.endValue, height:20, bottom:10 }
                ],
                series:series
            };
            // BCI：添加 DBCI / CBCI 背景色带
            if (isBCI) {
                var zones = getBCIMethodZones(facet.methods);
                if (zones.length > 0) {
                    option.series.push({
                        type: 'custom', name: '方法区域', silent: true, z: 0,
                        renderItem: function(params, api) {
                            var items = [];
                            var yMin = api.coord([0, Math.floor((yMinRaw-yPad)/5)*5]);
                            var yMax = api.coord([0, Math.ceil((yMaxRaw+yPad)/5)*5]);
                            zones.forEach(function(zone) {
                                var x1 = api.coord([zone.xMin, 0])[0];
                                var x2 = api.coord([zone.xMax, 0])[0];
                                items.push({
                                    type: 'rect', shape: { x: x1, y: yMax[1], width: x2 - x1, height: yMin[1] - yMax[1] },
                                    style: { fill: zone.color }, z2: 0
                                });
                                // 标签
                                items.push({
                                    type: 'text', x: (x1 + x2) / 2, y: yMax[1] + 14,
                                    style: { text: zone.name, fill: '#666', font: '11px sans-serif', textAlign: 'center' }, z2: 1
                                });
                            });
                            return { type: 'group', children: items };
                        },
                        data: [0]
                    });
                }
            }
            chart.setOption(option, true);
        }
        modeSelect.onchange = function(){ state.mode = modeSelect.value; applyAxisMode(); emitFacetState(); };
        chart.on('datazoom', function(){
            const zoom = (chart.getOption().dataZoom || [])[0] || {};
            if (zoom.startValue != null) state.startValue = zoom.startValue;
            if (zoom.endValue != null) state.endValue = zoom.endValue;
            emitFacetState();
        });
        chartDom.ondblclick = function(){ state.startValue=xMin; state.endValue=xMax; applyAxisMode(); emitFacetState(); };
        applyAxisMode();
        window._typeBCharts.push(chart);
        window.addEventListener('resize', function(){ chart.resize(); });
    });
}

function categoryZoom(categoryCount) {
    if (categoryCount <= 12) return [];
    const end = Math.max(1, Math.min(100, 1200 / categoryCount));
    return [
        { type:'inside', xAxisIndex:[0], start:0, end:end, zoomOnMouseWheel:false, moveOnMouseMove:true },
        { type:'slider', xAxisIndex:[0], start:0, end:end, height:20, bottom:8, minSpan:100/categoryCount }
    ];
}

function canonicalPulseId(value) {
    return String(value == null ? '' : value).trim().toLowerCase().replace(/\\s+/g, '');
}

function statusSummary(status) {
    if (!status || typeof status !== 'object') return '-';
    return Object.keys(status).sort().map(function(key){ return key+': '+displayValue(status[key]); }).join(' · ') || '-';
}

function renderTypeCComparison(typeCDatasets) {
    const container = document.getElementById('type-c-panel');
    if (!container) return;
    const pulseIds = [];
    const byDataset = typeCDatasets.map(function(ds){
        const map = {};
        ((ds.data && ds.data.pulses) || []).forEach(function(pulse){
            const key = canonicalPulseId(pulse.pulse_id);
            if (key && pulseIds.indexOf(key) < 0) pulseIds.push(key);
            map[key] = pulse;
        });
        return map;
    });
    pulseIds.sort(function(a,b){ return a.localeCompare(b, undefined, {numeric:true}); });
    const fields = [
        ['电压 (V)','voltage_V'], ['持续时间','duration'], ['持续时间单位','duration_unit'],
        ['内阻 (Ω)','internal_resistance_ohm'], ['测试等级','test_level'],
        ['脉冲次数','test_req_npulses'], ['脉冲间隔','test_req_interval'],
        ['试验持续时间','test_req_duration'], ['测试要求','test_req_raw'], ['功能状态','function_status']
    ];

    // 蝴蝶图：负脉冲向左，正脉冲向右
    // 正脉冲用暖色，负脉冲用冷色；标准A深色，标准B浅色
    let html = '<div class="type-c-view"><div class="comparison-heading"><h2>Type C 电源线瞬态脉冲对比</h2><span class="comparison-subtitle">蝴蝶图：左侧标准A ← 右侧标准B → · 暖色正脉冲 · 冷色负脉冲</span></div>';
    html += '<div class="type-c-chart" id="type-c-comparison-chart"></div>';
    html += '<div class="comparison-matrix-wrap"><table class="comparison-matrix"><thead><tr><th>Pulse ID</th><th>参数</th>';
    typeCDatasets.forEach(function(ds){ html += '<th>'+escapeHtml(ds.name)+'</th>'; });
    html += '</tr></thead><tbody>';
    pulseIds.forEach(function(pulseId){
        fields.forEach(function(field, fieldIdx){
            html += '<tr>';
            if (fieldIdx === 0) html += '<td rowspan="'+fields.length+'">'+escapeHtml(pulseId)+'</td>';
            html += '<td>'+escapeHtml(field[0])+'</td>';
            byDataset.forEach(function(map){
                const pulse = map[pulseId];
                const value = pulse ? (field[1] === 'function_status' ? statusSummary(pulse[field[1]]) : displayValue(pulse[field[1]])) : '-';
                html += '<td>'+escapeHtml(value)+'</td>';
            });
            html += '</tr>';
        });
    });
    html += '</tbody></table></div></div>';
    container.innerHTML = html;

    const chartDom = document.getElementById('type-c-comparison-chart');
    if (!chartDom || !pulseIds.length) return;
    const chart = echarts.init(chartDom);

    // 如果只有一个标准，使用简单柱状图回退（必须在所有 [1] 访问之前）
    if (typeCDatasets.length < 2) {
        var chartSeries = typeCDatasets.map(function(ds, dsIdx){
            const map = byDataset[dsIdx];
            return { name:ds.name, type:'bar', data:pulseIds.map(function(id){ const p=map[id]; return p && p.voltage_V != null ? Number(p.voltage_V) : null; }), itemStyle:{color:DATASET_PALETTE[dsIdx % DATASET_PALETTE.length]}, label:{show:true,position:'outside',fontSize:10} };
        });
        chart.setOption({
            tooltip:{trigger:'axis',axisPointer:{type:'shadow'}}, legend:{top:0},
            grid:{left:'6%',right:'3%',top:42,bottom:pulseIds.length>12?72:42,containLabel:true},
            xAxis:{type:'category',data:pulseIds,axisLabel:{rotate:pulseIds.length>8?30:0}},
            yAxis:{type:'value',name:'脉冲电压 (V)',nameLocation:'middle',nameGap:48},
            dataZoom:categoryZoom(pulseIds.length), series:chartSeries
        });
        window.addEventListener('resize', function(){ chart.resize(); });
        return;
    }

    // 蝴蝶图数据准备（2+ 标准）
    var warmColor = '#e74c3c';  // 正脉冲基准色
    var coolColor = '#3498db';  // 负脉冲基准色

    // 按脉冲 ID 获取电压，判断正负
    function getVoltage(dsIdx, pulseId) {
        var map = byDataset[dsIdx];
        var p = map[pulseId];
        return p && p.voltage_V != null ? Number(p.voltage_V) : null;
    }

    // 判断脉冲电压极性
    function pulsePolarity(pulseId) {
        for (var i = 0; i < byDataset.length; i++) {
            var v = getVoltage(i, pulseId);
            if (v != null) return v >= 0 ? 'positive' : 'negative';
        }
        return 'unknown';
    }

    // 标准A：向左延伸（负值）
    var seriesA = [];
    // 标准B：向右延伸（正值）
    var seriesB = [];

    pulseIds.forEach(function(pid) {
        var pol = pulsePolarity(pid);
        var baseColor = pol === 'positive' ? warmColor : coolColor;
        var vA = getVoltage(0, pid);
        var vB = getVoltage(1, pid);

        if (vA != null) {
            seriesA.push({
                name: typeCDatasets[0].name + ' ' + pid,
                value: -Math.abs(vA), // 标准A向左
                pulseId: pid, rawVoltage: vA, polarity: pol
            });
        }
        if (vB != null) {
            seriesB.push({
                name: typeCDatasets[1].name + ' ' + pid,
                value: Math.abs(vB), // 标准B向右
                pulseId: pid, rawVoltage: vB, polarity: pol
            });
        }
    });

    // 计算最大绝对值用于对称轴
    var maxAbs = 0;
    seriesA.forEach(function(d) { maxAbs = Math.max(maxAbs, Math.abs(d.value)); });
    seriesB.forEach(function(d) { maxAbs = Math.max(maxAbs, Math.abs(d.value)); });
    maxAbs = Math.ceil(maxAbs / 50) * 50 + 50;

    // 构建数据数组，长度与 pulseIds 一致，每个脉冲对应一个数据点（缺失填 null）
    var dataA = pulseIds.map(function(pid) {
        var v = getVoltage(0, pid);
        return v != null ? -Math.abs(v) : null;
    });
    var dataB = pulseIds.map(function(pid) {
        var v = getVoltage(1, pid);
        return v != null ? Math.abs(v) : null;
    });

    var chartSeries = [];

    // 标准A（左侧，负值）—— 单个 series，data 数组长度与 pulseIds 一致
    // 蝴蝶图：去掉 stack 避免正负值求和，用 barGap: '-100%' 让柱子从 0 轴重叠
    chartSeries.push({
        name: typeCDatasets[0].name,
        type: 'bar', barWidth: '55%',
        data: dataA.map(function(v, i) {
            if (v == null) return null;
            var pid = pulseIds[i];
            var pol = pulsePolarity(pid);
            var barColor = pol === 'positive' ? '#e74c3c' : '#2e86c1';
            return { value: v, pulseId: pid, itemStyle: { color: barColor, borderRadius: [4, 0, 0, 4] } };
        }),
        label: { show: true, position: 'left', fontSize: 10, formatter: function(p) { return p.value != null ? Math.abs(p.value) + 'V' : ''; } }
    });

    // 标准B（右侧，正值）—— 单个 series，data 数组长度与 pulseIds 一致
    chartSeries.push({
        name: typeCDatasets[1].name,
        type: 'bar', barWidth: '55%',
        data: dataB.map(function(v, i) {
            if (v == null) return null;
            var pid = pulseIds[i];
            var pol = pulsePolarity(pid);
            var barColor = pol === 'positive' ? '#e67e22' : '#1a5276';
            return { value: v, pulseId: pid, itemStyle: { color: barColor, borderRadius: [0, 4, 4, 0] } };
        }),
        label: { show: true, position: 'right', fontSize: 10, formatter: function(p) { return p.value != null ? Math.abs(p.value) + 'V' : ''; } }
    });

    // 蝴蝶图渲染
    chart.setOption({
            tooltip: {
                trigger: 'axis', axisPointer: { type: 'shadow' },
                formatter: function(params) {
                    var pid = '';
                    var html = '';
                    params.forEach(function(p) {
                        if (p.value == null) return;
                        pid = p.data.pulseId || pid;
                        var v = p.data.pulseId ? Math.abs(p.value) : p.value;
                        html += '<b>' + p.seriesName + '</b><br/>电压: ' + v + ' V<br/>';
                    });
                    if (pid) html = '<b>脉冲: ' + pid + '</b><br/>' + html;
                    return html || '无数据';
                }
            },
            legend: {
                top: 0,
                data: [typeCDatasets[0].name, typeCDatasets[1].name]
            },
            barGap: '-100%',  // 蝴蝶图关键：让两个标准的柱子从 0 轴重叠，负值向左、正值向右
            grid: { left: '5%', right: '5%', top: 48, bottom: 48, containLabel: true },
            xAxis: {
                type: 'value', min: -maxAbs, max: maxAbs,
                axisLabel: { formatter: function(v) { return Math.abs(v) + ' V'; } },
                splitLine: { lineStyle: { type: 'dashed', color: '#ddd' } },
                axisLine: { lineStyle: { color: '#999', width: 2 } }
            },
            yAxis: {
                type: 'category',
                data: pulseIds,
                axisLine: { show: false },
                axisTick: { show: false },
                position: 'center' // 标签居中
            },
            series: chartSeries
        });

    window.addEventListener('resize', function(){ chart.resize(); });
}

function normalizeDischargeType(value) {
    const text = String(value || '').trim().toLowerCase();
    if (text.indexOf('接触') >= 0 || text.indexOf('contact') >= 0 || text.indexOf('kontakt') >= 0) return '接触放电';
    if (text.indexOf('空气') >= 0 || text.indexOf('air') >= 0 || text.indexOf('luft') >= 0) return '空气放电';
    if (text.indexOf('间接') >= 0 || text.indexOf('indirect') >= 0 || text.indexOf('indirekt') >= 0) return '间接放电';
    return text || '未知放电类型';
}

function canonicalDischargeScenario(discharge) {
    const type = normalizeDischargeType(discharge.discharge_type);
    const points = String(discharge.discharge_points || '未指定场景').trim().toLowerCase().replace(/\\s+/g,' ');
    return type + '｜' + points;
}

// 构造雷达图指标：从 scenarios 中提取唯一的放电类型+场景，去重；同时返回 indicator→scenario 映射
function buildRadarIndicators(scenarios) {
    var seen = {};
    var indicators = [];
    var indicatorToScenario = {}; // indicator name → scenario key 直接映射
    scenarios.forEach(function(s) {
        var parts = s.split('｜');
        var type = parts[0] || '';
        var scene = (parts[1] || '').trim();
        // 截断场景描述，保留关键部分
        var shortScene = scene.length > 12 ? scene.slice(0, 12) + '...' : scene;
        var label = type + '\\n' + (shortScene || '');
        if (!seen[label]) {
            seen[label] = true;
            indicatorToScenario[label] = s;
            indicators.push({ name: label, max: 30 }); // 默认最大 30kV
        }
    });
    return { indicators: indicators, indicatorToScenario: indicatorToScenario };
}

function renderTypeDComparison(typeDDatasets) {
    const container = document.getElementById('type-d-panel');
    if (!container) return;
    const scenarios = [];
    const byDataset = typeDDatasets.map(function(ds){
        const map = {};
        ((ds.data && ds.data.discharges) || []).forEach(function(discharge){
            const key = canonicalDischargeScenario(discharge);
            if (scenarios.indexOf(key) < 0) scenarios.push(key);
            map[key] = discharge;
        });
        return map;
    });
    scenarios.sort();
    const fields = [['等级 (kV)','test_level_kV'],['HBM','human_body_model'],['放电点','discharge_points'],['放电次数','no_of_discharges'],['恢复时间','recovery_time'],['功能状态','function_status']];

    // 雷达图指标
    var radarResult = buildRadarIndicators(scenarios);
    var radarIndicators = radarResult.indicators;
    var indicatorToScenario = radarResult.indicatorToScenario; // 直接映射，无需反查
    // 动态调整 max 值
    var maxKV = 0;
    byDataset.forEach(function(map) {
        scenarios.forEach(function(key) {
            var d = map[key];
            if (d && d.test_level_kV != null) maxKV = Math.max(maxKV, Number(d.test_level_kV));
        });
    });
    var radarMax = Math.max(10, Math.ceil(maxKV / 5) * 5 + 5);
    radarIndicators.forEach(function(ind) { ind.max = radarMax; });

    // 布局：左侧雷达图 + 右侧参数对比表
    let html = '<div class="type-d-view"><div class="comparison-heading"><h2>Type D ESD 静电放电对比</h2><span class="comparison-subtitle">雷达图重叠对比 · 右侧表格查看细节</span></div>';
    html += '<div class="type-d-layout">';
    // 左侧雷达图
    html += '<div class="type-d-radar-wrap"><div class="type-d-chart" id="type-d-comparison-chart"></div>';
    // 雷达图图例
    html += '<div class="type-d-radar-legend">';
    typeDDatasets.forEach(function(ds, dsIdx) {
        html += '<span class="type-d-radar-legend-item"><span class="type-d-radar-legend-color" style="background:' + DATASET_PALETTE[dsIdx % DATASET_PALETTE.length] + '"></span>' + escapeHtml(ds.name) + '</span>';
    });
    html += '</div></div>';
    // 右侧参数表
    html += '<div class="type-d-table-wrap"><div class="comparison-matrix-wrap"><table class="comparison-matrix"><thead><tr><th>放电场景</th><th>参数</th>';
    typeDDatasets.forEach(function(ds){ html += '<th>'+escapeHtml(ds.name)+'</th>'; });
    html += '</tr></thead><tbody>';
    scenarios.forEach(function(scenario){
        fields.forEach(function(field, fieldIdx){
            html += '<tr>';
            if (fieldIdx === 0) html += '<td rowspan="'+fields.length+'">'+escapeHtml(scenario)+'</td>';
            html += '<td>'+escapeHtml(field[0])+'</td>';
            byDataset.forEach(function(map){
                const discharge = map[scenario];
                const value = discharge ? (field[1] === 'function_status' ? statusSummary(discharge[field[1]]) : displayValue(discharge[field[1]])) : '-';
                html += '<td>'+escapeHtml(value)+'</td>';
            });
            html += '</tr>';
        });
    });
    html += '</tbody></table></div></div>';
    html += '</div></div>';
    container.innerHTML = html;

    // ===== 雷达图渲染 =====
    const chartDom = document.getElementById('type-d-comparison-chart');
    if (!chartDom || !scenarios.length) return;
    const chart = echarts.init(chartDom);

    // 为每个标准构建雷达图数据，使用 indicatorToScenario 直接映射
    var radarSeries = typeDDatasets.map(function(ds, dsIdx) {
        var map = byDataset[dsIdx];
        var data = radarIndicators.map(function(ind) {
            var scenarioKey = indicatorToScenario[ind.name];
            if (scenarioKey) {
                var d = map[scenarioKey];
                return d && d.test_level_kV != null ? Number(d.test_level_kV) : 0;
            }
            return 0;
        });
        return {
            name: ds.name,
            type: 'radar',
            data: [{ value: data, name: ds.name }],
            symbol: 'circle',
            symbolSize: 5,
            lineStyle: { color: DATASET_PALETTE[dsIdx % DATASET_PALETTE.length], width: 2 },
            areaStyle: { color: DATASET_PALETTE[dsIdx % DATASET_PALETTE.length], opacity: 0.15 },
            itemStyle: { color: DATASET_PALETTE[dsIdx % DATASET_PALETTE.length] }
        };
    });

    chart.setOption({
        tooltip: { trigger: 'item' },
        legend: { bottom: 0, data: typeDDatasets.map(function(ds) { return ds.name; }) },
        radar: {
            center: ['50%', '52%'],
            radius: '65%',
            indicator: radarIndicators,
            axisName: { fontSize: 10, color: '#555' },
            splitArea: { areaStyle: { color: ['rgba(200,200,200,0.05)', 'rgba(200,200,200,0.1)'] } }
        },
        series: radarSeries
    });
    window.addEventListener('resize', function(){ chart.resize(); });
}

function downloadPNG(){
    const chart = window._chart;
    if (!chart) { alert('请先生成图表'); return; }
    const url = chart.getDataURL({ type: 'png', pixelRatio: 2 });
    const a = document.createElement('a');
    a.href = url;
    a.download = 'emc_chart_' + Date.now() + '.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function resetZoom(){
    const chart = window._chart;
    if (!chart) { alert('请先生成图表'); return; }
    const cfg = emcData && emcData.chart_config;
    applyTypeARange(chart, cfg && cfg.x_union ? cfg.x_union : typeARangeFromSeries(window._series || []));
}

function renderOverview(datasets) {
    const container = document.getElementById('overview-panel');
    if (!container) return;

    // 收集所有测试项目的摘要信息
    var overviewRows = [];

    // Type_A：发射类
    var typeA = datasets.filter(function(d){ return (d.type || 'TypeA') === 'TypeA'; });
    typeA.forEach(function(ds) {
        var bands = (ds.data && ds.data.bands) || [];
        var limits = [];
        bands.forEach(function(band) {
            var freq = band.frequency || {};
            var freqStr = (freq.start || '') + '~' + (freq.end || '') + 'MHz';
            var dets = Object.keys(band.limits || {});
            dets.forEach(function(det) {
                var lim = band.limits[det];
                var val = (lim && lim.value != null) ? lim.value : (lim && lim.formula ? lim.formula : '-');
                limits.push(freqStr + ' ' + det + ': ' + val);
            });
        });
        overviewRows.push({
            category: '发射类',
            item: ds.name || '辐射/传导发射',
            summary: limits.slice(0, 3).join('; ') + (limits.length > 3 ? '...' : ''),
            rawData: ds
        });
    });

    // Type_B：抗扰度类
    var typeB = datasets.filter(function(d){ return d.type === 'type_b'; });
    typeB.forEach(function(ds) {
        var methods = (ds.data && ds.data.test_methods) || [];
        var items = [];
        methods.forEach(function(m) {
            var family = m.method_family || '其他';
            var val = m.value != null ? m.value + (m.unit || '') : (m.formula || '-');
            items.push(family + ': ' + val + ' @' + m.frequency);
        });
        overviewRows.push({
            category: '抗扰度类',
            item: ds.name || '抗扰度',
            summary: items.slice(0, 3).join('; ') + (items.length > 3 ? '...' : ''),
            rawData: ds
        });
    });

    // Type_C：瞬态脉冲类
    var typeC = datasets.filter(function(d){ return d.type === 'type_c'; });
    typeC.forEach(function(ds) {
        var pulses = (ds.data && ds.data.pulses) || [];
        var items = [];
        pulses.forEach(function(p) {
            items.push('脉冲' + p.pulse_id + ': ' + (p.voltage_V != null ? p.voltage_V + 'V' : '-'));
        });
        overviewRows.push({
            category: '瞬态脉冲类',
            item: ds.name || '瞬态脉冲',
            summary: items.join('; '),
            rawData: ds
        });
    });

    // Type_D：ESD类
    var typeD = datasets.filter(function(d){ return d.type === 'type_d'; });
    typeD.forEach(function(ds) {
        var discharges = (ds.data && ds.data.discharges) || [];
        var items = [];
        discharges.forEach(function(d) {
            var type = normalizeDischargeType(d.discharge_type);
            items.push(type + ': ' + (d.test_level_kV != null ? d.test_level_kV + 'kV' : '-'));
        });
        overviewRows.push({
            category: 'ESD类',
            item: ds.name || 'ESD',
            summary: items.join('; '),
            rawData: ds
        });
    });

    if (overviewRows.length === 0) return;

    // 按 category 分组
    var categories = [];
    var seenCat = {};
    overviewRows.forEach(function(row) {
        if (!seenCat[row.category]) {
            seenCat[row.category] = true;
            categories.push(row.category);
        }
    });

    // 获取数据集名称列表作为列
    var dsNames = datasets.map(function(ds) { return ds.name || 'Unknown'; });

    // 构建对比表：行 = 测试项目，列 = 各标准的值
    // 按 category 分组的行
    var html = '<div class="overview-wrap"><h2 style="font-size:18px;font-weight:600;margin-bottom:12px;">综合对比总览</h2>';
    html += '<div class="overview-legend">';
    html += '<span class="overview-legend-item"><span class="overview-legend-swatch" style="background:#f8d7da"></span> 标准A更严</span>';
    html += '<span class="overview-legend-item"><span class="overview-legend-swatch" style="background:#fff3cd"></span> 相同</span>';
    html += '<span class="overview-legend-item"><span class="overview-legend-swatch" style="background:#d4edda"></span> 标准B更严</span>';
    html += '<span class="overview-legend-item"><span class="overview-legend-swatch" style="background:#ecf0f1"></span> 无法比较</span>';
    html += '</div>';

    html += '<table class="overview-table"><thead><tr><th>测试类别</th><th>测试项目</th>';
    dsNames.forEach(function(name) { html += '<th>' + escapeHtml(name) + '</th>'; });
    html += '<th>对比结论</th></tr></thead><tbody>';

    // 判断严酷度：对于发射/抗扰度，限值越低 = 越严
    function getSeverity(rowA, rowB) {
        if (!rowA || !rowB) return 'na';
        // 提取关键数值进行比较
        function extractKeyValue(row) {
            var ds = row.rawData;
            if ((ds.type || 'TypeA') === 'TypeA') {
                var bands = (ds.data && ds.data.bands) || [];
                var minVal = Infinity;
                bands.forEach(function(b) {
                    var dets = Object.keys(b.limits || {});
                    dets.forEach(function(det) {
                        var lim = b.limits[det];
                        if (lim && lim.value != null) minVal = Math.min(minVal, Number(lim.value));
                    });
                });
                return isFinite(minVal) ? minVal : null;
            }
            if (ds.type === 'type_b') {
                var methods = (ds.data && ds.data.test_methods) || [];
                var minVal = Infinity;
                methods.forEach(function(m) {
                    if (m.value != null) minVal = Math.min(minVal, Number(m.value));
                });
                return isFinite(minVal) ? minVal : null;
            }
            if (ds.type === 'type_c') {
                var pulses = (ds.data && ds.data.pulses) || [];
                var maxAbs = 0;
                pulses.forEach(function(p) {
                    if (p.voltage_V != null) maxAbs = Math.max(maxAbs, Math.abs(Number(p.voltage_V)));
                });
                return maxAbs > 0 ? maxAbs : null;
            }
            if (ds.type === 'type_d') {
                var discharges = (ds.data && ds.data.discharges) || [];
                var maxKV = 0;
                discharges.forEach(function(d) {
                    if (d.test_level_kV != null) maxKV = Math.max(maxKV, Number(d.test_level_kV));
                });
                return maxKV > 0 ? maxKV : null;
            }
            return null;
        }

        var valA = extractKeyValue(rowA);
        var valB = extractKeyValue(rowB);
        if (valA == null || valB == null) return 'na';
        if (Math.abs(valA - valB) < 0.01) return 'same';
        // 发射/抗扰度：限值越低越严；ESD/瞬态：值越高越严
        var cat = rowA.category;
        if (cat === 'ESD类' || cat === '瞬态脉冲类') {
            return valA > valB ? 'a' : 'b';
        }
        return valA < valB ? 'a' : 'b';
    }

    var severityLabels = { a: '标准A更严', b: '标准B更严', same: '相近', na: '无法比较' };
    var severityClasses = { a: 'severity-a', b: 'severity-b', same: 'severity-same', na: 'severity-na' };

    categories.forEach(function(cat) {
        var catRows = overviewRows.filter(function(r) { return r.category === cat; });
        // 按名称对齐两个标准
        var rowA = catRows[0];
        var rowB = catRows.length > 1 ? catRows[1] : null;
        var severity = rowB ? getSeverity(rowA, rowB) : 'na';

        html += '<tr>';
        html += '<td>' + escapeHtml(cat) + '</td>';
        html += "<td class=\\\"overview-clickable\\\" onclick=\\\"scrollToCategory('" + escapeHtml(cat) + "')\\\">" + escapeHtml(rowA.item) + "</td>";
        html += '<td>' + escapeHtml(rowA.summary) + '</td>';
        if (rowB) {
            html += '<td>' + escapeHtml(rowB.summary) + '</td>';
        } else {
            html += '<td class="severity-na">-</td>';
        }
        html += '<td class="' + severityClasses[severity] + '">' + severityLabels[severity] + '</td>';
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// 滚动到对应类别
function scrollToCategory(cat) {
    var panel = document.getElementById('type-b-panel');
    if (panel) {
        var sections = panel.querySelectorAll('.type-b-facet');
        sections.forEach(function(section) {
            var title = (section.querySelector('.type-b-facet-title') || {}).textContent || '';
            if (title.indexOf(cat) >= 0 || cat.indexOf(title) >= 0) {
                section.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    }
}
function init(){
    const datasets = emcData.datasets || [];
    const typeA = datasets.filter(function(d){ return (d.type || 'TypeA') === 'TypeA'; });
    const typeB = datasets.filter(function(d){ return d.type === 'type_b'; });
    const typeC = datasets.filter(function(d){ return d.type === 'type_c'; });
    const typeD = datasets.filter(function(d){ return d.type === 'type_d'; });

    const toolbar = document.getElementById('toolbar');
    if (toolbar) toolbar.style.display = typeA.length > 0 ? 'flex' : 'none';
    // Bind toolbar buttons via id (avoids onclick-in-HTML which needs global functions)
    const btnExport = document.getElementById('btn-download-png');
    const btnReset = document.getElementById('btn-reset-zoom');
    if (btnExport) { btnExport.onclick = downloadPNG; }
    if (btnReset) { btnReset.onclick = resetZoom; }

    // Show/hide panels based on type
    const chartEl = document.getElementById('chart');
    const ctrlPanel = document.getElementById('control-panel');
    if (typeA.length > 0) {
        if (chartEl) chartEl.style.display = '';
        if (ctrlPanel) ctrlPanel.style.display = '';
        renderTypeA(typeA);
    } else {
        if (chartEl) chartEl.style.display = 'none';
        if (ctrlPanel) ctrlPanel.style.display = 'none';
    }

    if (typeB.length > 0) renderTypeBFacets(typeB);
    if (typeC.length > 0) renderTypeCComparison(typeC);
    if (typeD.length > 0) renderTypeDComparison(typeD);
}

init();
