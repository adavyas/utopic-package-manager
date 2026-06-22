import * as fs from "node:fs";
import * as http from "node:http";
import * as https from "node:https";
import * as os from "node:os";
import * as path from "node:path";
import * as readline from "node:readline";
import { spawn, type ChildProcess } from "node:child_process";

const VERSION = "0.1.4";

type ModelEntry = {
  id: string;
  name: string;
  family: string;
  filename: string;
  url: string;
  size: string;
  recommended: boolean;
  description: string;
};

type ChatOptions = {
  model: string | null;
  server: string | null;
  host: string;
  port: string;
  ngl: string;
  ctxSize: string;
  maxTokens: number;
  temperature: number;
  help: boolean;
  version: boolean;
};

type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

function printHelp(): void {
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
  /clear                Clear this session's conversation.
  /system TEXT          Set or replace the system prompt.
  /exit                 Quit.

Examples:
  utopic chat
  utopic chat dream-7b-q4
  utopic chat -m /path/to/model.gguf -ngl 99
  utopic chat --server http://127.0.0.1:8910
`);
}

function parseArgs(argv: string[]): ChatOptions {
  const options: ChatOptions = {
    model: null,
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
  const positional: string[] = [];
  let modelArgs = 0;
  const valueAfterEquals = (arg: string, flag: string): string => arg.slice(flag.length + 1);
  const looksLikeNegativeNumber = (value: string): boolean => value.length > 1 && value[0] === "-" && /\d/.test(value[1]);
  const requiredValue = (flag: string, value: string, allowNegativeNumber = false): string => {
    if (value === "" || (value.startsWith("-") && !(allowNegativeNumber && looksLikeNegativeNumber(value)))) {
      throw new Error(`expected a value after ${flag}`);
    }
    return value;
  };
  const numberValue = (flag: string, value: string): number => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) throw new Error(`${flag} must be a number`);
    return parsed;
  };
  const positiveIntegerValue = (flag: string, value: string): number => {
    const parsed = numberValue(flag, value);
    if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${flag} must be a positive integer`);
    return parsed;
  };
  const nonNegativeNumberValue = (flag: string, value: string): number => {
    const parsed = numberValue(flag, value);
    if (parsed < 0) throw new Error(`${flag} must be a non-negative number`);
    return parsed;
  };
  const integerString = (flag: string, value: string, min: number, max: number | null, label: string): string => {
    const parsed = Number(value);
    if (!Number.isInteger(parsed) || parsed < min || (max !== null && parsed > max)) {
      throw new Error(`${flag} must be ${label}`);
    }
    return value;
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = (flag = arg, allowNegativeNumber = false): string => {
      if (i + 1 >= argv.length) throw new Error(`expected a value after ${arg}`);
      i += 1;
      return requiredValue(flag, argv[i], allowNegativeNumber);
    };
    if (arg === "-h" || arg === "--help") options.help = true;
    else if (arg === "--version") options.version = true;
    else if (arg === "-m" || arg === "--model") {
      modelArgs += 1;
      options.model = next("-m/--model");
    }
    else if (arg.startsWith("--model=")) {
      modelArgs += 1;
      options.model = requiredValue("-m/--model", valueAfterEquals(arg, "--model"));
    }
    else if (arg === "--server") options.server = next("--server");
    else if (arg.startsWith("--server=")) options.server = requiredValue("--server", valueAfterEquals(arg, "--server"));
    else if (arg === "--host") options.host = next("--host");
    else if (arg.startsWith("--host=")) options.host = requiredValue("--host", valueAfterEquals(arg, "--host"));
    else if (arg === "--port") options.port = integerString("--port", next("--port", true), 1, 65535, "an integer from 1 to 65535");
    else if (arg.startsWith("--port=")) options.port = integerString("--port", requiredValue("--port", valueAfterEquals(arg, "--port"), true), 1, 65535, "an integer from 1 to 65535");
    else if (arg === "-ngl") options.ngl = integerString("-ngl", next("-ngl", true), 0, null, "a non-negative integer");
    else if (arg === "--ctx-size") options.ctxSize = integerString("--ctx-size", next("--ctx-size", true), 1, null, "a positive integer");
    else if (arg.startsWith("--ctx-size=")) options.ctxSize = integerString("--ctx-size", requiredValue("--ctx-size", valueAfterEquals(arg, "--ctx-size"), true), 1, null, "a positive integer");
    else if (arg === "--max-tokens") options.maxTokens = positiveIntegerValue("--max-tokens", next("--max-tokens", true));
    else if (arg.startsWith("--max-tokens=")) options.maxTokens = positiveIntegerValue("--max-tokens", requiredValue("--max-tokens", valueAfterEquals(arg, "--max-tokens"), true));
    else if (arg === "--temperature") options.temperature = nonNegativeNumberValue("--temperature", next("--temperature", true));
    else if (arg.startsWith("--temperature=")) options.temperature = nonNegativeNumberValue("--temperature", requiredValue("--temperature", valueAfterEquals(arg, "--temperature"), true));
    else if (arg === "--no-setup") continue;
    else if (arg.startsWith("-")) throw new Error(`unknown option: ${arg}`);
    else positional.push(arg);
  }
  if (modelArgs + positional.length > 1) throw new Error("expected at most one model argument");
  if (!options.model && positional.length > 0) options.model = positional[0];
  return options;
}

function catalogPath(): string {
  return process.env.UTOPIC_MODELS_CATALOG ?? path.resolve(__dirname, "..", "models.json");
}

function cacheRoot(): string {
  return process.env.UTOPIC_HOME ?? path.join(os.homedir(), ".cache", "utopic");
}

function modelsDir(): string {
  return process.env.UTOPIC_MODELS_DIR ?? path.join(cacheRoot(), "models");
}

function binDir(): string {
  return process.env.UTOPIC_BIN_DIR ?? path.join(cacheRoot(), "bin");
}

function serverBinary(): string {
  return path.join(binDir(), process.platform === "win32" ? "utopic_server.exe" : "utopic_server");
}

function serverLogPath(): string {
  return process.env.UTOPIC_SERVER_LOG ?? path.join(cacheRoot(), "utopic-server.log");
}

function clientHost(host: string): string {
  return host === "0.0.0.0" || host === "::" || host === "" ? "127.0.0.1" : host;
}

function httpClientForUrl(parsed: URL, label: string): typeof http | typeof https {
  if (parsed.protocol === "https:") return https;
  if (parsed.protocol === "http:") return http;
  throw new Error(`${label} must use http:// or https://`);
}

function normalizeServerBaseUrl(value: string): string {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error("--server must be a URL");
  }
  httpClientForUrl(parsed, "--server");
  if (parsed.pathname.replace(/\/+$/, "") === "/v1/chat/completions") {
    parsed.pathname = "/";
    parsed.search = "";
    parsed.hash = "";
  }
  return parsed.toString().replace(/\/+$/, "");
}

function chatCompletionsUrl(baseUrl: string): string {
  return new URL("/v1/chat/completions", baseUrl).toString();
}

function readCatalog(): ModelEntry[] {
  const file = catalogPath();
  let data: unknown;
  try {
    data = JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    throw new Error(`Failed to read model catalog ${file}: ${(error as Error).message}`);
  }
  if (!Array.isArray(data)) throw new Error(`Model catalog ${file} must contain a JSON list`);
  if (data.length === 0) throw new Error("Utopic model catalog is empty");
  return data.map((item, index) => validateCatalogEntry(item, index));
}

function validateCatalogEntry(item: unknown, index: number): ModelEntry {
  if (item === null || typeof item !== "object" || Array.isArray(item)) {
    throw new Error(`Invalid model catalog entry ${index}: expected a JSON object`);
  }
  const entry = item as Partial<ModelEntry>;
  for (const field of ["id", "name", "family", "filename", "url", "size", "description"] as const) {
    if (typeof entry[field] !== "string") {
      throw new Error(`Invalid model catalog entry ${index}: ${field} must be a string`);
    }
  }
  if (typeof entry.recommended !== "boolean") {
    throw new Error(`Invalid model catalog entry ${index}: recommended must be a boolean`);
  }
  return entry as ModelEntry;
}

function safeModelFilename(entry: ModelEntry): string {
  if (
    !entry.filename ||
    entry.filename === "." ||
    entry.filename === ".." ||
    entry.filename.includes("/") ||
    entry.filename.includes("\\") ||
    entry.filename.includes(":")
  ) {
    throw new Error(`unsafe model filename for '${entry.id}': ${entry.filename}`);
  }
  return entry.filename;
}

function validateModelUrl(entry: ModelEntry): void {
  let parsed: URL;
  try {
    parsed = new URL(entry.url);
  } catch {
    throw new Error(`model URL for '${entry.id}' must be a URL`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(`unsupported model URL protocol for '${entry.id}': ${parsed.protocol || "<missing>"}`);
  }
  if (!parsed.host) {
    throw new Error(`model URL for '${entry.id}' must include a host`);
  }
}

function localModelPath(entry: ModelEntry): string {
  return path.join(modelsDir(), safeModelFilename(entry));
}

function parseContentLength(value: string | undefined): number {
  if (!value) return 0;
  const total = Number(value);
  if (!Number.isInteger(total) || total < 0) throw new Error(`invalid content-length: ${value}`);
  return total;
}

function normalizeDownloadError(error: Error): Error {
  if (/content-length/i.test(error.message) && /parse error|invalid/i.test(error.message)) {
    return new Error("invalid content-length");
  }
  return error;
}

function isLikelyPath(value: string): boolean {
  return value.includes("/") || value.includes("\\") || value.toLowerCase().endsWith(".gguf");
}

function resolveLocalPath(value: string): string {
  if (value === "~") return os.homedir();
  if (value.startsWith("~/") || value.startsWith("~\\")) return path.join(os.homedir(), value.slice(2));
  return path.resolve(value);
}

function ask(rl: readline.Interface, text: string): Promise<string> {
  return new Promise((resolve) => rl.question(text, resolve));
}

async function chooseModel(catalog: ModelEntry[]): Promise<string> {
  const recommended = catalog.find((entry) => entry.recommended) ?? catalog[0];
  if (!process.stdin.isTTY) return recommended.id;

  console.log("\nAvailable models:");
  catalog.forEach((entry, index) => {
    const marker = entry.recommended ? "*" : " ";
    const modelPath = localModelPath(entry);
    const exists = fs.existsSync(modelPath) && fs.statSync(modelPath).size > 0 ? "downloaded" : "not downloaded";
    console.log(`${index + 1}. ${marker} ${entry.id} (${entry.size}, ${exists})`);
    console.log(`   ${entry.name}`);
  });

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  try {
    const answer = (await ask(rl, `\nChoose a model [${recommended.id}]: `)).trim();
    if (!answer) return recommended.id;
    const numeric = Number(answer);
    if (Number.isInteger(numeric) && numeric >= 1 && numeric <= catalog.length) {
      return catalog[numeric - 1].id;
    }
    return answer;
  } finally {
    rl.close();
  }
}

async function resolveModel(value: string | null): Promise<string> {
  if (value && isLikelyPath(value)) return resolveLocalPath(value);

  const catalog = readCatalog();
  const modelId = value ?? await chooseModel(catalog);
  const entry = catalog.find((item) => item.id === modelId);
  if (!entry) throw new Error(`unknown model '${modelId}'. Run 'utopic models list' to see aliases.`);
  const destination = localModelPath(entry);
  if (fs.existsSync(destination) && fs.statSync(destination).size > 0) return destination;

  validateModelUrl(entry);
  console.log(`\nPulling ${entry.name} from Hugging Face`);
  console.log(entry.url);
  return download(entry.url, destination);
}

function download(url: string, destination: string, redirectsRemaining = 10): Promise<string> {
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  const partial = `${destination}.partial`;
  const removeEmptyDestinationOnFailure =
    fs.existsSync(destination) && fs.statSync(destination).size === 0;
  if (fs.existsSync(partial)) fs.unlinkSync(partial);

  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const client = parsed.protocol === "https:" ? https : parsed.protocol === "http:" ? http : null;
    let settled = false;
    const removePartial = (): void => {
      if (fs.existsSync(partial)) fs.unlinkSync(partial);
    };
    const removeStaleDestination = (): void => {
      if (
        removeEmptyDestinationOnFailure &&
        fs.existsSync(destination) &&
        fs.statSync(destination).size === 0
      ) {
        fs.unlinkSync(destination);
      }
    };
    const fail = (error: Error): void => {
      if (settled) return;
      settled = true;
      removePartial();
      removeStaleDestination();
      reject(error);
    };
    const succeed = (value: string): void => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    if (!client) {
      fail(new Error(`unsupported download protocol: ${parsed.protocol}`));
      return;
    }

    const request = client.get(parsed, (response: http.IncomingMessage) => {
      if (response.statusCode && response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        response.resume();
        if (redirectsRemaining <= 0) {
          fail(new Error("too many model download redirects"));
          return;
        }
        const nextUrl = new URL(response.headers.location, parsed).toString();
        download(nextUrl, destination, redirectsRemaining - 1).then(succeed, fail);
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
      const incompleteDownloadError = (): Error => new Error(`downloaded ${downloaded} of ${expectedTotal} bytes`);
      const out = fs.createWriteStream(partial);
      response.on("data", (chunk: Buffer) => {
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
          if (expectedTotal) process.stdout.write("\n");
          try {
            if (downloaded === 0) throw new Error("downloaded 0 bytes");
            if (expectedTotal && downloaded !== expectedTotal) throw incompleteDownloadError();
            fs.renameSync(partial, destination);
            succeed(destination);
          } catch (renameError) {
            fail(renameError as Error);
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

function waitForHealth(baseUrl: string, timeoutMs: number, shouldStop?: () => boolean): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  const healthUrl = new URL("/health", baseUrl);
  const client = httpClientForUrl(healthUrl, "--server");
  return new Promise((resolve, reject) => {
    const retry = (): void => {
      if (shouldStop?.()) return;
      if (Date.now() > deadline) {
        reject(new Error(`timed out waiting for ${healthUrl.toString()}`));
        return;
      }
      setTimeout(attempt, 300);
    };
    const attempt = (): void => {
      if (shouldStop?.()) return;
      const req = client.get(healthUrl, (res: http.IncomingMessage) => {
        res.resume();
        if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) resolve();
        else retry();
      });
      req.on("error", retry);
    };
    attempt();
  });
}

async function startServer(options: ChatOptions, modelPath: string): Promise<{ baseUrl: string; child: ChildProcess }> {
  const binary = serverBinary();
  if (!fs.existsSync(binary)) throw new Error("Utopic native binaries are missing. Run `utopic setup`, then retry.");
  const baseUrl = `http://${clientHost(options.host)}:${options.port}`;
  const logPath = serverLogPath();
  fs.mkdirSync(path.dirname(logPath), { recursive: true });
  const log = fs.openSync(logPath, "a");
  const child = spawn(binary, [
    "-m", modelPath,
    "--host", options.host,
    "--port", options.port,
    "-ngl", options.ngl,
    "--ctx-size", options.ctxSize,
  ], { stdio: ["ignore", log, log], detached: false });
  let waitingForHealth = true;
  const earlyExit = new Promise<never>((_, reject) => {
    child.once("error", (error) => {
      if (waitingForHealth) reject(error);
    });
    child.once("exit", (code, signal) => {
      if (waitingForHealth) {
        const status = code === null ? `signal ${signal}` : `code ${code}`;
        reject(new Error(`utopic-server exited before it became healthy (${status}). Logs: ${logPath}`));
      }
    });
  });
  try {
    await Promise.race([waitForHealth(baseUrl, 120000, () => !waitingForHealth), earlyExit]);
  } finally {
    waitingForHealth = false;
  }
  console.log(`\nOpenAI-compatible URL: ${baseUrl}/v1/chat/completions`);
  console.log(`Server logs: ${logPath}\n`);
  return { baseUrl, child };
}

function requestJson(url: string, body: unknown): Promise<any> {
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
    }, (res: http.IncomingMessage) => {
      let data = "";
      res.setEncoding("utf8");
      res.on("data", (chunk: string) => { data += chunk; });
      res.on("end", () => {
        if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
          reject(new Error(`HTTP ${res.statusCode}: ${data}`));
          return;
        }
        resolve(JSON.parse(data));
      });
    });
    req.on("error", reject);
    req.write(payload);
    req.end();
  });
}

async function chatLoop(baseUrl: string, options: ChatOptions): Promise<void> {
  const interactive = process.stdin.isTTY;
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: interactive ? "utopic> " : "" });
  const messages: ChatMessage[] = [];
  console.log("Type /help for commands. Type /exit to quit.\n");
  if (interactive) rl.prompt();
  for await (const line of rl) {
    const input = line.trim();
    if (!input) {
      if (interactive) rl.prompt();
      continue;
    }
    if (input === "/exit" || input === "/quit") break;
    if (input === "/clear") {
      messages.length = 0;
      console.log("conversation cleared");
      if (interactive) rl.prompt();
      continue;
    }
    if (input === "/help") {
      console.log("/clear        clear conversation history");
      console.log("/system TEXT  set or replace the system prompt");
      console.log("/exit         quit");
      if (interactive) rl.prompt();
      continue;
    }
    if (input.startsWith("/system ")) {
      const content = input.slice("/system ".length).trim();
      const existing = messages.find((message) => message.role === "system");
      if (existing) existing.content = content;
      else messages.unshift({ role: "system", content });
      console.log("system prompt updated");
      if (interactive) rl.prompt();
      continue;
    }

    messages.push({ role: "user", content: input });
    process.stdout.write("assistant> ");
    try {
      const response = await requestJson(chatCompletionsUrl(baseUrl), {
        model: "utopic",
        messages,
        max_tokens: options.maxTokens,
        temperature: options.temperature,
      });
      const content = response.choices?.[0]?.message?.content ?? "";
      console.log(String(content).trim());
      messages.push({ role: "assistant", content: String(content) });
    } catch (error) {
      messages.pop();
      console.error(`\nrequest failed: ${(error as Error).message}`);
    }
    if (interactive) rl.prompt();
  }
  rl.close();
}

async function main(): Promise<number> {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return 0;
  }
  if (options.version) {
    console.log(`utopic chat ${VERSION}`);
    return 0;
  }

  let child: ChildProcess | null = null;
  let baseUrl = options.server;
  try {
    if (!baseUrl) {
      const modelPath = await resolveModel(options.model);
      const started = await startServer(options, modelPath);
      baseUrl = started.baseUrl;
      child = started.child;
    } else {
      baseUrl = normalizeServerBaseUrl(baseUrl);
      await waitForHealth(baseUrl, 10000);
      console.log(`OpenAI-compatible URL: ${chatCompletionsUrl(baseUrl)}`);
    }
    await chatLoop(baseUrl, options);
    return 0;
  } finally {
    if (child && !child.killed) child.kill("SIGTERM");
  }
}

main().then((code) => {
  process.exitCode = code;
}).catch((error: Error) => {
  console.error(`utopic chat: ${error.message}`);
  process.exitCode = 1;
});
