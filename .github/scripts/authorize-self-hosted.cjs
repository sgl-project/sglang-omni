// SPDX-License-Identifier: Apache-2.0
const {readFileSync} = require('node:fs');

const repository = 'sgl-project/sglang-omni';
const hardwareWorkflows = ['omni-ci', 'omni-cpu-ci', 'omni-xpu-ci', 'mac-metal-ci'];

module.exports = async function authorize(github, context, environment) {
  const {owner, repo} = context.repo;
  const caller = environment.GITHUB_WORKFLOW_REF;
  if (`${owner}/${repo}` !== repository) {
    throw new Error('Self-hosted CI is restricted to the upstream repository.');
  }

  if (context.eventName === 'workflow_dispatch') {
    const mainWorkflow = hardwareWorkflows.some(name =>
      caller === `${repository}/.github/workflows/${name}.yaml@refs/heads/main`);
    if (!mainWorkflow || context.ref !== 'refs/heads/main') {
      throw new Error('Dispatch a hardware workflow on protected main.');
    }
    return;
  }

  const eventPr = context.payload.pull_request;
  if (context.eventName !== 'pull_request' || !eventPr) {
    throw new Error('Expected a pull request event.');
  }
  const pull_number = eventPr.number;
  const path = hardwareWorkflows.map(name => `.github/workflows/${name}.yaml`).find(path =>
    caller === `${repository}/${path}@refs/pull/${pull_number}/merge`);
  if (!path || !/^[a-f0-9]{40}$/.test(environment.GITHUB_WORKFLOW_SHA || '')) {
    throw new Error('Expected a hardware CI entry workflow on a pull request.');
  }

  // A directly called model suite also receives inputs from its caller.
  const {data: wrapper} = await github.rest.repos.getContent({
    owner, repo, path, ref: environment.GITHUB_WORKFLOW_SHA,
  });
  if (wrapper.type !== 'file' || wrapper.encoding !== 'base64' ||
      !readFileSync(path).equals(Buffer.from(wrapper.content, 'base64'))) {
    throw new Error('The PR entry workflow must match protected main.');
  }

  const {data: pr} = await github.rest.pulls.get({owner, repo, pull_number});
  const sha = eventPr.head.sha;
  if (pr.state !== 'open' || pr.draft || pr.base.ref !== 'main' || pr.head.sha !== sha) {
    throw new Error('Expected an open, ready PR targeting main at the original head SHA.');
  }
  const trustedIds = (environment.TRUSTED_USER_IDS || '').split(/[\s,]+/);
  if (pr.user.type === 'User' && trustedIds.includes(String(pr.user.id)) &&
      (pr.head.repo?.full_name === repository || pr.head.repo?.owner.id === pr.user.id)) {
    return;
  }

  const reviews = await github.paginate(github.rest.pulls.listReviews, {owner, repo, pull_number, per_page: 100});
  const decisions = new Map();
  // GitHub returns reviews chronologically. Comments do not replace a decision.
  for (const review of reviews) {
    if (review.user?.type === 'User' && review.user.id !== pr.user.id &&
        ['APPROVED', 'CHANGES_REQUESTED', 'DISMISSED'].includes(review.state)) {
      decisions.set(review.user.id, review);
    }
  }
  for (const review of decisions.values()) {
    if (review.state === 'APPROVED' && review.commit_id === sha) {
      const {data} = await github.rest.repos.getCollaboratorPermissionLevel({
        owner, repo, username: review.user.login,
      });
      if (data.user.id === review.user.id && ['admin', 'write', 'maintain'].includes(data.permission)) {
        console.log(`PR #${pull_number} at ${sha} approved by @${review.user.login}.`);
        return;
      }
    }
  }
  throw new Error(`Review and approve PR #${pull_number} at ${sha}, then re-run the workflow.`);
};
