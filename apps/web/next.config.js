/** @type {import('next').NextConfig} */
const nextConfig = {};

// Linux/Docker 用 standalone 输出；Windows 本地构建可设 NEXT_DISABLE_STANDALONE=1 规避符号链接权限问题
if (process.env.NEXT_DISABLE_STANDALONE !== "1" && process.platform !== "win32") {
  nextConfig.output = "standalone";
}

module.exports = nextConfig;
