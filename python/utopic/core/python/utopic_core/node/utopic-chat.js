"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const fs = require("node:fs");
const http = require("node:http");
const https = require("node:https");
const os = require("node:os");
const path = require("node:path");
const readline = require("node:readline");
const node_child_process_1 = require("node:child_process");
const VERSION = "0.1.8";
const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer naturally, directly, and with useful specifics. " +
    "When asked what you can do, give a concise overview with examples across writing, research, coding, math, planning, and conversation. " +
    "Do not claim access to the internet, private files, or real-world actions unless the user provides tools or context.";
function printHelp() {
    console.log(`usage: utopic chat [model-alias|/path/to/model.gguf] [options]

Start an Ollama-style terminal chat backed by the local Utopic server.

Options:
  -m, --model VALUE     Model alias or GGUF path.
  --server URL          Connect to an existing OpenAI-compatible Utopic server.
  --host HOST           Host for an auto-started server. Default: 127.0.0.1
  --port PORT           Port for an auto-started server. Default: 8910
  -ngl N                GPU layers for an auto-started server. Default: 99
  --ctx-size N          Context size for an auto-started server. Default: 4096
  --max-tokens N        Max response tokens. Default: 512
  --temperature N       Sampling temperature. Default: 0
  --no-setup            Skip Python-side first-use setup.
  -h, --help            Show this help.

Chat commands:
  /help                 Show chat commands.
  /models               Show catalog models with native readiness.
  /pull MODEL           Download a native text model. Restart chat to switch.
  /serve                Show local OpenAI-compatible endpoints.
  /clear                Clear this session's conversation.
  /system TEXT          Set or replace the system prompt.
  /exit                 Quit.

Examples:
  utopic chat
  utopic chat diffusiongemma-26b-a4b-q4
  utopic chat -m /path/to/model.gguf -ngl 99
  utopic chat --server http://127.0.0.1:8910
`);
}
function parseArgs(argv) {
    const options = {
        model: null,
        requestModel: "utopic",
        server: null,
        host: "127.0.0.1",
        port: "8910",
        ngl: "99",
        ctxSize: "4096",
        maxTokens: 512,
        temperature: 0,
        help: false,
        version: false,
    };
    const positional = [];
    let modelArgs = 0;
    const valueAfterEquals = (arg, flag) => arg.slice(flag.length + 1);
    const looksLikeNegativeNumber = (value) => value.length > 1 && value[0] === "-" && /\d/.test(value[1]);
    const requiredValue = (flag, value, allowNegativeNumber = false) => {
        if (value === "" || (value.startsWith("-") && !(allowNegativeNumber && looksLikeNegativeNumber(value)))) {
            throw new Error(`expected a value after ${flag}`);
        }
        return value;
    };
    const numberValue = (flag, value) => {
        const parsed = Number(value);
        if (!Number.isFinite(parsed))
            throw new Error(`${flag} must be a number`);
        return parsed;
    };
    const positiveIntegerValue = (flag, value) => {
        const parsed = numberValue(flag, value);
        if (!Number.isInteger(parsed) || parsed < 1)
            throw new Error(`${flag} must be a positive integer`);
        return parsed;
    };
    const nonNegativeNumberValue = (flag, value) => {
        const parsed = numberValue(flag, value);
        if (parsed < 0)
            throw new Error(`${flag} must be a non-negative number`);
        return parsed;
    };
    const integerString = (flag, value, min, max, label) => {
        const parsed = Number(value);
        if (!Number.isInteger(parsed) || parsed < min || (max !== null && parsed > max)) {
            throw new Error(`${flag} must be ${label}`);
        }
        return value;
    };
    for (let i = 0; i < argv.length; i += 1) {
        const arg = argv[i];
        const next = (flag = arg, allowNegativeNumber = false) => {
            if (i + 1 >= argv.length)
                throw new Error(`expected a value after ${arg}`);
            i += 1;
            return requiredValue(flag, argv[i], allowNegativeNumber);
        };
        if (arg === "-h" || arg === "--help")
            options.help = true;
        else if (arg === "--version")
            options.version = true;
        else if (arg === "-m" || arg === "--model") {
            modelArgs += 1;
            options.model = next("-m/--model");
        }
        else if (arg.startsWith("--model=")) {
            modelArgs += 1;
            options.model = requiredValue("-m/--model", valueAfterEquals(arg, "--model"));
        }
        else if (arg === "--server")
            options.server = next("--server");
        else if (arg.startsWith("--server="))
            options.server = requiredValue("--server", valueAfterEquals(arg, "--server"));
        else if (arg === "--host")
            options.host = next("--host");
        else if (arg.startsWith("--host="))
            options.host = requiredValue("--host", valueAfterEquals(arg, "--host"));
        else if (arg === "--port")
            options.port = integerString("--port", next("--port", true), 1, 65535, "an integer from 1 to 65535");
        else if (arg.startsWith("--port="))
            options.port = integerString("--port", requiredValue("--port", valueAfterEquals(arg, "--port"), true), 1, 65535, "an integer from 1 to 65535");
        else if (arg === "-ngl")
            options.ngl = integerString("-ngl", next("-ngl", true), 0, null, "a non-negative integer");
        else if (arg === "--ctx-size")
            options.ctxSize = integerString("--ctx-size", next("--ctx-size", true), 1, null, "a positive integer");
        else if (arg.startsWith("--ctx-size="))
            options.ctxSize = integerString("--ctx-size", requiredValue("--ctx-size", valueAfterEquals(arg, "--ctx-size"), true), 1, null, "a positive integer");
        else if (arg === "--max-tokens")
            options.maxTokens = positiveIntegerValue("--max-tokens", next("--max-tokens", true));
        else if (arg.startsWith("--max-tokens="))
            options.maxTokens = positiveIntegerValue("--max-tokens", requiredValue("--max-tokens", valueAfterEquals(arg, "--max-tokens"), true));
        else if (arg === "--temperature")
            options.temperature = nonNegativeNumberValue("--temperature", next("--temperature", true));
        else if (arg.startsWith("--temperature="))
            options.temperature = nonNegativeNumberValue("--temperature", requiredValue("--temperature", valueAfterEquals(arg, "--temperature"), true));
        else if (arg === "--no-setup")
            continue;
        else if (arg.startsWith("-"))
            throw new Error(`unknown option: ${arg}`);
        else
            positional.push(arg);
    }
    if (modelArgs + positional.length > 1)
        throw new Error("expected at most one model argument");
    if (!options.model && positional.length > 0)
        options.model = positional[0];
    return options;
}
function configuredPath(name, fallback) {
    const value = process.env[name];
    return value ? resolveLocalPath(value) : fallback;
}
function catalogPath() {
    return configuredPath("UTOPIC_MODELS_CATALOG", path.resolve(__dirname, "..", "models.json"));
}
function cacheRoot() {
    return configuredPath("UTOPIC_HOME", path.join(os.homedir(), ".cache", "utopic"));
}
function modelsDir() {
    return configuredPath("UTOPIC_MODELS_DIR", path.join(cacheRoot(), "models"));
}
function binDir() {
    return configuredPath("UTOPIC_BIN_DIR", path.join(cacheRoot(), "bin"));
}
function runnerBinary() {
    return path.join(binDir(), process.platform === "win32" ? "utopic_runner.exe" : "utopic_runner");
}
function cliBinary() {
    if (process.env.UTOPIC_CLI)
        return resolveLocalPath(process.env.UTOPIC_CLI);
    const binary = path.join(binDir(), process.platform === "win32" ? "utopic.exe" : "utopic");
    return fs.existsSync(binary) ? binary : "utopic";
}
function requireRunnerBinary() {
    const binary = runnerBinary();
    if (!fs.existsSync(binary))
        throw new Error("Utopic native binaries are missing. Run `utopic setup`, then retry.");
    return binary;
}
function serverLogPath() {
    return configuredPath("UTOPIC_SERVER_LOG", path.join(cacheRoot(), "utopic-server.log"));
}
function clientHost(host) {
    return host === "0.0.0.0" || host === "::" || host === "" ? "127.0.0.1" : host;
}
function httpClientForUrl(parsed, label) {
    if (parsed.protocol === "https:")
        return https;
    if (parsed.protocol === "http:")
        return http;
    throw new Error(`${label} must use http:// or https://`);
}
function normalizeServerBaseUrl(value) {
    let parsed;
    try {
        parsed = new URL(value);
    }
    catch {
        throw new Error("--server must be a URL");
    }
    httpClientForUrl(parsed, "--server");
    const pathname = parsed.pathname.replace(/\/+$/, "");
    if (pathname.endsWith("/v1/chat/completions")) {
        parsed.pathname = pathname.slice(0, -"/v1/chat/completions".length) || "/";
        parsed.search = "";
        parsed.hash = "";
    }
    else if (pathname.endsWith("/v1")) {
        parsed.pathname = pathname.slice(0, -"/v1".length) || "/";
        parsed.search = "";
        parsed.hash = "";
    }
    return parsed.toString().replace(/\/+$/, "");
}
function joinServerPath(baseUrl, suffix) {
    const parsed = new URL(baseUrl);
    const basePath = parsed.pathname.replace(/\/+$/, "");
    const suffixPath = suffix.replace(/^\/+/, "");
    parsed.pathname = basePath ? `${basePath}/${suffixPath}` : `/${suffixPath}`;
    parsed.search = "";
    parsed.hash = "";
    return parsed.toString();
}
function chatCompletionsUrl(baseUrl) {
    const pathname = new URL(baseUrl).pathname.replace(/\/+$/, "");
    if (pathname.endsWith("/v1"))
        return joinServerPath(baseUrl, "chat/completions");
    return joinServerPath(baseUrl, "v1/chat/completions");
}
function readCatalog() {
    const file = catalogPath();
    let data;
    try {
        data = JSON.parse(fs.readFileSync(file, "utf8"));
    }
    catch (error) {
        throw new Error(`Failed to read model catalog ${file}: ${error.message}`);
    }
    if (!Array.isArray(data))
        throw new Error(`Model catalog ${file} must contain a JSON list`);
    if (data.length === 0)
        throw new Error("Utopic model catalog is empty");
    return data.map((item, index) => validateCatalogEntry(item, index));
}
function validateCatalogEntry(item, index) {
    if (item === null || typeof item !== "object" || Array.isArray(item)) {
        throw new Error(`Invalid model catalog entry ${index}: expected a JSON object`);
    }
    const entry = item;
    for (const field of ["id", "name", "family", "filename", "url", "size", "description"]) {
        if (typeof entry[field] !== "string") {
            throw new Error(`Invalid model catalog entry ${index}: ${field} must be a string`);
        }
    }
    if (typeof entry.recommended !== "boolean") {
        throw new Error(`Invalid model catalog entry ${index}: recommended must be a boolean`);
    }
    if (entry.bytes !== undefined &&
        (!Number.isInteger(entry.bytes) || entry.bytes <= 0)) {
        throw new Error(`Invalid model catalog entry ${index}: bytes must be a positive integer`);
    }
    for (const field of ["hardware", "supported_backends", "endpoints", "outputs"]) {
        const value = entry[field];
        if (value !== undefined &&
            (!Array.isArray(value) || value.length === 0 || value.some((part) => typeof part !== "string" || part.length === 0))) {
            throw new Error(`Invalid model catalog entry ${index}: ${field} must be a non-empty string list`);
        }
    }
    if (entry.modality !== undefined && !["text", "image", "tts", "music", "video", "misc"].includes(entry.modality)) {
        throw new Error(`Invalid model catalog entry ${index}: modality is not supported`);
    }
    if (entry.runtime !== undefined && !["native", "bridge"].includes(entry.runtime)) {
        throw new Error(`Invalid model catalog entry ${index}: runtime is not supported`);
    }
    if (entry.native_status !== undefined &&
        !["ready", "planned", "experimental", "unsupported_on_device"].includes(entry.native_status)) {
        throw new Error(`Invalid model catalog entry ${index}: native_status is not supported`);
    }
    for (const field of ["expected_vram_gib", "expected_ram_gib"]) {
        const value = entry[field];
        if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value) || value <= 0)) {
            throw new Error(`Invalid model catalog entry ${index}: ${field} must be a positive number`);
        }
    }
    return entry;
}
function modelModality(entry) {
    return entry.modality ?? "text";
}
function modelRuntime(entry) {
    return entry.runtime ?? "native";
}
function modelNativeStatus(entry) {
    return entry.native_status ?? (modelRuntime(entry) === "bridge" ? "planned" : "ready");
}
function modelRunner(entry) {
    return entry.runner ?? (modelRuntime(entry) === "bridge" ? `${modelModality(entry)}_runner` : "utopic_runner");
}
function modelBackends(entry) {
    return (entry.supported_backends && entry.supported_backends.length > 0 ? entry.supported_backends : ["metal", "cuda", "cpu"]).join(", ");
}
function modelMemory(entry) {
    const parts = [];
    if (entry.expected_vram_gib !== undefined)
        parts.push(`VRAM ${entry.expected_vram_gib} GiB`);
    if (entry.expected_ram_gib !== undefined)
        parts.push(`RAM ${entry.expected_ram_gib} GiB`);
    return parts.length > 0 ? parts.join(", ") : "memory TBD";
}
function isNativeTextModel(entry) {
    return modelModality(entry) === "text" && modelRuntime(entry) === "native";
}
function printModelCatalog(catalog, options) {
    const visible = options.chatOnly ? catalog.filter(isNativeTextModel) : catalog;
    if (visible.length === 0) {
        console.log(options.chatOnly ? "No native text models are available in this catalog." : "No models are available in this catalog.");
        return;
    }
    console.log(options.chatOnly ? "\nAvailable chat models:" : "\nCatalog models:");
    visible.forEach((entry, index) => {
        const marker = entry.recommended ? "*" : " ";
        const downloaded = isModelDownloaded(entry) ? "downloaded" : "not downloaded";
        console.log(`${index + 1}. ${marker} ${entry.id} (${entry.size}, ${downloaded})`);
        console.log(`   ${entry.name}`);
        console.log(`   ${modelModality(entry)} / ${modelRuntime(entry)} / ${modelNativeStatus(entry)} / ${modelRunner(entry)}`);
        console.log(`   backends: ${modelBackends(entry)}; ${modelMemory(entry)}`);
    });
}
function safeModelFilename(entry) {
    if (!entry.filename ||
        entry.filename === "." ||
        entry.filename === ".." ||
        entry.filename.includes("/") ||
        entry.filename.includes("\\") ||
        entry.filename.includes(":")) {
        throw new Error(`unsafe model filename for '${entry.id}': ${entry.filename}`);
    }
    return entry.filename;
}
function validateModelUrl(entry) {
    let parsed;
    try {
        parsed = new URL(entry.url);
    }
    catch {
        throw new Error(`model URL for '${entry.id}' must be a URL`);
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        throw new Error(`unsupported model URL protocol for '${entry.id}': ${parsed.protocol || "<missing>"}`);
    }
    if (!parsed.host) {
        throw new Error(`model URL for '${entry.id}' must include a host`);
    }
}
function localModelPath(entry) {
    if (entry.runtime === "bridge")
        return path.join(modelsDir(), entry.id);
    return path.join(modelsDir(), safeModelFilename(entry));
}
function parseContentLength(value) {
    if (!value)
        return 0;
    const total = Number(value);
    if (!Number.isInteger(total) || total < 0)
        throw new Error(`invalid content-length: ${value}`);
    return total;
}
function isNonEmptyFile(filePath) {
    if (!fs.existsSync(filePath))
        return false;
    const stats = fs.statSync(filePath);
    return stats.isFile() && stats.size > 0;
}
function isInteractiveInput() {
    return Boolean(process.stdin.isTTY || process.env.UTOPIC_CHAT_FORCE_TTY === "1");
}
function isModelDownloaded(entry) {
    const filePath = localModelPath(entry);
    if (entry.runtime === "bridge")
        return fs.existsSync(path.join(filePath, "utopic-model.json"));
    if (!isNonEmptyFile(filePath))
        return false;
    if (entry.bytes === undefined)
        return true;
    return fs.statSync(filePath).size === entry.bytes;
}
function isEmptyFile(filePath) {
    if (!fs.existsSync(filePath))
        return false;
    const stats = fs.statSync(filePath);
    return stats.isFile() && stats.size === 0;
}
function removePathIfExists(filePath) {
    if (fs.existsSync(filePath))
        fs.rmSync(filePath, { recursive: true, force: true });
}
function normalizeDownloadError(error) {
    if (/content-length/i.test(error.message) && /parse error|invalid/i.test(error.message)) {
        return new Error("invalid content-length");
    }
    return error;
}
function isLikelyPath(value) {
    return value.includes("/") || value.includes("\\") || value.toLowerCase().endsWith(".gguf");
}
function resolveLocalPath(value) {
    if (value === "~")
        return os.homedir();
    if (value.startsWith("~/") || value.startsWith("~\\"))
        return path.join(os.homedir(), value.slice(2));
    return path.resolve(value);
}
function ask(rl, text) {
    return new Promise((resolve) => rl.question(text, resolve));
}
async function chooseModel(catalog) {
    const chatModels = catalog.filter(isNativeTextModel);
    const recommended = chatModels.find((entry) => entry.recommended) ?? chatModels[0];
    if (!recommended)
        throw new Error("no native text models are available in this catalog");
    if (!isInteractiveInput())
        return recommended.id;
    printModelCatalog(catalog, { chatOnly: true });
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    try {
        const answer = (await ask(rl, `\nChoose a model [${recommended.id}]: `)).trim();
        if (!answer)
            return recommended.id;
        const numeric = Number(answer);
        if (Number.isInteger(numeric) && numeric >= 1 && numeric <= chatModels.length) {
            return chatModels[numeric - 1].id;
        }
        return answer;
    }
    finally {
        rl.close();
    }
}
async function resolveModel(value, beforeDownload) {
    if (value && isLikelyPath(value))
        return resolveLocalPath(value);
    const catalog = readCatalog();
    const modelId = value ?? await chooseModel(catalog);
    const entry = catalog.find((item) => item.id === modelId);
    if (!entry)
        throw new Error(`unknown model '${modelId}'. Run 'utopic models list' to see aliases.`);
    if (!isNativeTextModel(entry)) {
        throw new Error(`model '${modelId}' is ${modelModality(entry)} / ${modelRuntime(entry)} / ${modelNativeStatus(entry)}; ` +
            "use `utopic serve` or the runtime gateway for non-chat modalities.");
    }
    const destination = localModelPath(entry);
    if (isModelDownloaded(entry))
        return destination;
    validateModelUrl(entry);
    beforeDownload?.();
    console.log(`\nPulling ${entry.name} from Hugging Face`);
    console.log(entry.url);
    return download(entry.url, destination, 10, entry.bytes);
}
async function resolveChatModel(value, beforeDownload) {
    if (value && isLikelyPath(value)) {
        const resolvedPath = await resolveModel(value, beforeDownload);
        return { runModel: resolvedPath, requestModel: "utopic", resolvedPath };
    }
    const catalog = readCatalog();
    const modelId = value ?? await chooseModel(catalog);
    const resolvedPath = await resolveModel(modelId, beforeDownload);
    return { runModel: modelId, requestModel: modelId, resolvedPath };
}
function download(url, destination, redirectsRemaining = 10, expectedBytes) {
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    const partial = `${destination}.partial`;
    const removeEmptyDestinationOnFailure = isEmptyFile(destination);
    if (fs.existsSync(destination) && !fs.statSync(destination).isFile())
        removePathIfExists(destination);
    removePathIfExists(partial);
    return new Promise((resolve, reject) => {
        const parsed = new URL(url);
        const client = parsed.protocol === "https:" ? https : parsed.protocol === "http:" ? http : null;
        let settled = false;
        const removePartial = () => {
            removePathIfExists(partial);
        };
        const removeStaleDestination = () => {
            if (removeEmptyDestinationOnFailure &&
                isEmptyFile(destination)) {
                removePathIfExists(destination);
            }
        };
        const fail = (error) => {
            if (settled)
                return;
            settled = true;
            removePartial();
            removeStaleDestination();
            reject(error);
        };
        const succeed = (value) => {
            if (settled)
                return;
            settled = true;
            resolve(value);
        };
        if (!client) {
            fail(new Error(`unsupported download protocol: ${parsed.protocol}`));
            return;
        }
        const request = client.get(parsed, (response) => {
            if (response.statusCode && response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
                response.resume();
                if (redirectsRemaining <= 0) {
                    fail(new Error("too many model download redirects"));
                    return;
                }
                const nextUrl = new URL(response.headers.location, parsed).toString();
                download(nextUrl, destination, redirectsRemaining - 1, expectedBytes).then(succeed, fail);
                return;
            }
            if (response.statusCode !== 200) {
                response.resume();
                fail(new Error(`HTTP ${response.statusCode}`));
                return;
            }
            const contentLength = Array.isArray(response.headers["content-length"])
                ? response.headers["content-length"][0]
                : response.headers["content-length"];
            const expectedTotal = parseContentLength(contentLength);
            let downloaded = 0;
            const incompleteDownloadError = () => new Error(`downloaded ${downloaded} of ${expectedTotal} bytes`);
            const out = fs.createWriteStream(partial);
            response.on("data", (chunk) => {
                downloaded += chunk.length;
                if (expectedTotal) {
                    const percent = String(Math.floor((downloaded * 100) / expectedTotal)).padStart(3, " ");
                    process.stdout.write(`\rDownloading ${path.basename(destination)}: ${percent}%`);
                }
            });
            response.pipe(out);
            out.on("finish", () => {
                out.close((error) => {
                    if (error) {
                        fail(error);
                        return;
                    }
                    if (expectedTotal)
                        process.stdout.write("\n");
                    try {
                        if (downloaded === 0)
                            throw new Error("downloaded 0 bytes");
                        if (expectedTotal && downloaded !== expectedTotal)
                            throw incompleteDownloadError();
                        if (expectedBytes !== undefined && downloaded !== expectedBytes) {
                            throw new Error(`downloaded ${downloaded} of ${expectedBytes} bytes`);
                        }
                        fs.renameSync(partial, destination);
                        succeed(destination);
                    }
                    catch (renameError) {
                        fail(renameError);
                    }
                });
            });
            response.on("error", (error) => fail(normalizeDownloadError(error)));
            response.on("aborted", () => fail(expectedTotal ? incompleteDownloadError() : new Error("download aborted")));
            out.on("error", fail);
        });
        request.on("error", (error) => fail(normalizeDownloadError(error)));
    });
}
function waitForHealth(baseUrl, timeoutMs, shouldStop) {
    const deadline = Date.now() + timeoutMs;
    const healthUrl = new URL(joinServerPath(baseUrl, "health"));
    const client = httpClientForUrl(healthUrl, "--server");
    return new Promise((resolve, reject) => {
        const retry = () => {
            if (shouldStop?.())
                return;
            if (Date.now() > deadline) {
                reject(new Error(`timed out waiting for ${healthUrl.toString()}`));
                return;
            }
            setTimeout(attempt, 300);
        };
        const attempt = () => {
            if (shouldStop?.())
                return;
            const req = client.get(healthUrl, (res) => {
                res.resume();
                if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300)
                    resolve();
                else
                    retry();
            });
            req.on("error", retry);
        };
        attempt();
    });
}
async function startServer(options, runModel) {
    const baseUrl = `http://${clientHost(options.host)}:${options.port}`;
    const logPath = serverLogPath();
    fs.mkdirSync(path.dirname(logPath), { recursive: true });
    const log = fs.openSync(logPath, "a");
    const child = (0, node_child_process_1.spawn)(cliBinary(), [
        "run",
        runModel,
        "--host", options.host,
        "--port", options.port,
        "-ngl", options.ngl,
        "--ctx-size", options.ctxSize,
        "--no-setup",
    ], { stdio: ["ignore", log, log], detached: false });
    let waitingForHealth = true;
    const earlyExit = new Promise((_, reject) => {
        child.once("error", (error) => {
            if (waitingForHealth)
                reject(error);
        });
        child.once("exit", (code, signal) => {
            if (waitingForHealth) {
                const status = code === null ? `signal ${signal}` : `code ${code}`;
                reject(new Error(`utopic run exited before it became healthy (${status}). Logs: ${logPath}`));
            }
        });
    });
    try {
        await Promise.race([waitForHealth(baseUrl, 120000, () => !waitingForHealth), earlyExit]);
    }
    finally {
        waitingForHealth = false;
    }
    console.log(`\nOpenAI-compatible URL: ${baseUrl}/v1/chat/completions`);
    console.log(`Server logs: ${logPath}\n`);
    return { baseUrl, child };
}
function requestJson(url, body) {
    const parsed = new URL(url);
    const client = httpClientForUrl(parsed, "request URL");
    const payload = JSON.stringify(body);
    return new Promise((resolve, reject) => {
        const req = client.request({
            method: "POST",
            hostname: parsed.hostname,
            port: parsed.port,
            path: parsed.pathname,
            headers: {
                "content-type": "application/json",
                "content-length": Buffer.byteLength(payload),
            },
        }, (res) => {
            let data = "";
            res.setEncoding("utf8");
            res.on("data", (chunk) => { data += chunk; });
            res.on("end", () => {
                if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
                    reject(new Error(`HTTP ${res.statusCode}: ${data}`));
                    return;
                }
                try {
                    resolve(JSON.parse(data));
                }
                catch {
                    reject(new Error("invalid JSON response"));
                }
            });
        });
        req.on("error", reject);
        req.write(payload);
        req.end();
    });
}
function findSseBoundary(buffer) {
    const lf = buffer.indexOf("\n\n");
    const crlf = buffer.indexOf("\r\n\r\n");
    if (lf < 0 && crlf < 0)
        return null;
    if (lf >= 0 && (crlf < 0 || lf < crlf))
        return { index: lf, length: 2 };
    return { index: crlf, length: 4 };
}
function chatCompletionText(payload) {
    if (!payload || typeof payload !== "object")
        return "";
    const choices = payload.choices;
    if (!Array.isArray(choices) || choices.length === 0)
        return "";
    const first = choices[0];
    if (!first || typeof first !== "object")
        return "";
    const message = first.message;
    if (!message || typeof message !== "object")
        return "";
    const content = message.content;
    return typeof content === "string" ? content : "";
}
function requestChatCompletionStream(url, body, onContent) {
    const parsed = new URL(url);
    const client = httpClientForUrl(parsed, "request URL");
    const payload = JSON.stringify({ ...body, stream: true });
    return new Promise((resolve, reject) => {
        let content = "";
        let buffer = "";
        const req = client.request({
            method: "POST",
            hostname: parsed.hostname,
            port: parsed.port,
            path: parsed.pathname,
            headers: {
                "content-type": "application/json",
                "content-length": Buffer.byteLength(payload),
                "accept": "text/event-stream",
            },
        }, (res) => {
            if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
                let data = "";
                res.setEncoding("utf8");
                res.on("data", (chunk) => { data += chunk; });
                res.on("end", () => reject(new Error(`HTTP ${res.statusCode}: ${data}`)));
                return;
            }
            const contentType = String(res.headers["content-type"] ?? "").toLowerCase();
            if (!contentType.includes("text/event-stream")) {
                let data = "";
                res.setEncoding("utf8");
                res.on("data", (chunk) => { data += chunk; });
                res.on("end", () => {
                    try {
                        const text = chatCompletionText(JSON.parse(data));
                        if (text.length > 0)
                            onContent(text);
                        resolve(text);
                    }
                    catch {
                        reject(new Error("invalid JSON chat completion response"));
                    }
                });
                return;
            }
            res.setEncoding("utf8");
            res.on("data", (chunk) => {
                buffer += chunk;
                let boundary = findSseBoundary(buffer);
                while (boundary) {
                    const event = buffer.slice(0, boundary.index);
                    buffer = buffer.slice(boundary.index + boundary.length);
                    boundary = findSseBoundary(buffer);
                    for (const line of event.split(/\r?\n/)) {
                        if (!line.startsWith("data:"))
                            continue;
                        const data = line.slice("data:".length).trimStart();
                        if (!data || data === "[DONE]")
                            continue;
                        try {
                            const parsedEvent = JSON.parse(data);
                            const delta = parsedEvent.choices?.[0]?.delta?.content;
                            if (typeof delta === "string" && delta.length > 0) {
                                content += delta;
                                onContent(delta);
                            }
                        }
                        catch {
                            reject(new Error("invalid SSE chat completion chunk"));
                            req.destroy();
                            return;
                        }
                    }
                }
            });
            res.on("end", () => resolve(content));
        });
        req.on("error", reject);
        req.write(payload);
        req.end();
    });
}
async function chatLoop(baseUrl, options) {
    const interactive = isInteractiveInput();
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: interactive ? ">>> " : "" });
    const messages = [{ role: "system", content: DEFAULT_SYSTEM_PROMPT }];
    console.log("Type /help for commands, /models for catalog, /serve for endpoints, /exit to quit.\n");
    if (interactive)
        rl.prompt();
    for await (const line of rl) {
        const input = line.trim();
        if (!input) {
            if (interactive)
                rl.prompt();
            continue;
        }
        if (input === "/exit" || input === "/quit")
            break;
        if (input === "/clear") {
            messages.length = 0;
            messages.push({ role: "system", content: DEFAULT_SYSTEM_PROMPT });
            console.log("Conversation cleared.");
            if (interactive)
                rl.prompt();
            continue;
        }
        if (input === "/help") {
            console.log("/models       Show catalog models with native readiness.");
            console.log("/pull MODEL   Download a native text model. Restart chat to switch.");
            console.log("/serve        Show local OpenAI-compatible endpoints.");
            console.log("/clear        Clear conversation history.");
            console.log("/system TEXT  Set or replace the system prompt.");
            console.log("/exit         Quit.");
            if (interactive)
                rl.prompt();
            continue;
        }
        if (input === "/models") {
            try {
                printModelCatalog(readCatalog(), { chatOnly: false });
            }
            catch (error) {
                console.error(`catalog failed: ${error.message}`);
            }
            if (interactive)
                rl.prompt();
            continue;
        }
        if (input === "/serve") {
            console.log(`Chat completions: ${chatCompletionsUrl(baseUrl)}`);
            console.log(`Models: ${joinServerPath(baseUrl, "v1/models")}`);
            console.log(`MCP: ${joinServerPath(baseUrl, "mcp")}`);
            if (interactive)
                rl.prompt();
            continue;
        }
        if (input.startsWith("/pull ")) {
            const model = input.slice("/pull ".length).trim();
            if (!model) {
                console.error("usage: /pull MODEL");
            }
            else {
                try {
                    const modelPath = await resolveModel(model);
                    console.log(`Pulled ${model} to ${modelPath}`);
                    console.log("Restart chat with that model to switch the running server.");
                }
                catch (error) {
                    console.error(`pull failed: ${error.message}`);
                }
            }
            if (interactive)
                rl.prompt();
            continue;
        }
        if (input.startsWith("/system ")) {
            const content = input.slice("/system ".length).trim();
            const existing = messages.find((message) => message.role === "system");
            if (existing)
                existing.content = content;
            else
                messages.unshift({ role: "system", content });
            console.log("System prompt updated.");
            if (interactive)
                rl.prompt();
            continue;
        }
        messages.push({ role: "user", content: input });
        try {
            const body = {
                model: options.requestModel,
                messages,
                max_tokens: options.maxTokens,
                temperature: options.temperature,
            };
            let content = "";
            if (interactive) {
                rl.pause();
                process.stdout.write("Thinking...\n");
                content = await requestChatCompletionStream(chatCompletionsUrl(baseUrl), body, (delta) => {
                    process.stdout.write(delta);
                });
                process.stdout.write("\n\n");
                rl.resume();
            }
            else {
                const response = await requestJson(chatCompletionsUrl(baseUrl), body);
                content = String(response.choices?.[0]?.message?.content ?? "").trim();
                console.log(content);
            }
            messages.push({ role: "assistant", content });
        }
        catch (error) {
            messages.pop();
            if (interactive)
                rl.resume();
            console.error(`\nrequest failed: ${error.message}`);
        }
        if (interactive)
            rl.prompt();
    }
    rl.close();
}
async function main() {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) {
        printHelp();
        return 0;
    }
    if (options.version) {
        console.log(`utopic chat ${VERSION}`);
        return 0;
    }
    let child = null;
    let baseUrl = options.server;
    try {
        if (!baseUrl) {
            let runnerBinaryPath = null;
            const ensureRunnerBinary = () => {
                runnerBinaryPath ?? (runnerBinaryPath = requireRunnerBinary());
                return runnerBinaryPath;
            };
            const selected = await resolveChatModel(options.model, ensureRunnerBinary);
            ensureRunnerBinary();
            options.requestModel = selected.requestModel;
            const started = await startServer(options, selected.runModel);
            baseUrl = started.baseUrl;
            child = started.child;
        }
        else {
            baseUrl = normalizeServerBaseUrl(baseUrl);
            await waitForHealth(baseUrl, 10000);
            console.log(`OpenAI-compatible URL: ${chatCompletionsUrl(baseUrl)}`);
        }
        await chatLoop(baseUrl, options);
        return 0;
    }
    finally {
        if (child && !child.killed)
            child.kill("SIGTERM");
    }
}
main().then((code) => {
    process.exitCode = code;
}).catch((error) => {
    console.error(`utopic chat: ${error.message}`);
    process.exitCode = 1;
});
