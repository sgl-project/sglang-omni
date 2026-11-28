// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const test = require('node:test');
const authorize = require('../../../.github/scripts/authorize-self-hosted.cjs');

const repo = 'sgl-project/sglang-omni';
const wrapperPath = '.github/workflows/omni-ci.yaml';
const originalHead = 'a'.repeat(40);
const nextHead = 'b'.repeat(40);
const author = {id: 10, login: 'contributor', type: 'User'};
const reviewer = {id: 20, login: 'maintainer', type: 'User'};
const approval = {state: 'APPROVED', commit_id: originalHead, user: reviewer};

function check({head = originalHead, eventHead = originalHead, reviews = [], permission = 'write',
                trusted = '', event = 'pull_request', caller, changedWrapper = false,
                state = 'open', draft = false, ref = 'refs/heads/main', headOwner = author} = {}) {
  const pr = {number: 1, state, draft, user: author, base: {ref: 'main'},
    head: {sha: head, repo: {owner: headOwner, full_name: headOwner.login + '/fork'}}};
  const github = {rest: {pulls: {get: async () => ({data: pr}), listReviews: 'reviews'}, repos: {
    getCollaboratorPermissionLevel: async () => ({data: {permission, user: reviewer}}),
    getContent: async ({path}) => ({data: {type: 'file', encoding: 'base64',
      content: (changedWrapper ? Buffer.from('different') : readFileSync(path)).toString('base64')}}),
  }}, paginate: async () => reviews};
  const context = {repo: {owner: 'sgl-project', repo: 'sglang-omni'}, ref, eventName: event,
    payload: {pull_request: {number: 1, head: {sha: eventHead}}}};
  const environment = {GITHUB_WORKFLOW_REF: caller || repo + '/' + wrapperPath + '@refs/pull/1/merge',
    GITHUB_WORKFLOW_SHA: originalHead, TRUSTED_USER_IDS: trusted};
  return authorize(github, context, environment);
}

test('a later push and an old event each need a fresh decision on the live head', async () => {
  await assert.rejects(check(), /Review and approve/);
  await check({reviews: [approval]});
  await assert.rejects(check({head: nextHead, reviews: [approval]}), /original head SHA/);
  await assert.rejects(check({head: nextHead, eventHead: nextHead, reviews: [approval]}), /Review and approve/);
  await check({head: nextHead, eventHead: nextHead, reviews: [{...approval, commit_id: nextHead}]});
});

test('only an effective review from an independent human with current write access counts', async () => {
  for (const state of ['DISMISSED', 'CHANGES_REQUESTED']) {
    await assert.rejects(check({reviews: [approval, {...approval, state}]}), /Review and approve/);
  }
  for (const user of [author, {...reviewer, type: 'Bot'}]) {
    await assert.rejects(check({reviews: [{...approval, user}]}), /Review and approve/);
  }
  await assert.rejects(check({permission: 'read', reviews: [approval]}), /Review and approve/);
  await check({reviews: [approval, {...approval, state: 'COMMENTED'}]});
});

test('trusted authors still need a valid entry workflow, ready PR and qualifying head repository', async () => {
  const trusted = String(author.id);
  await check({trusted});
  await assert.rejects(check({trusted, headOwner: reviewer}), /Review and approve/);
  await assert.rejects(check({trusted, state: 'closed'}), /open, ready/);
  await assert.rejects(check({trusted, draft: true}), /open, ready/);
  await assert.rejects(check({trusted, changedWrapper: true}), /match protected main/);
  await assert.rejects(check({trusted, caller: repo + '/.github/workflows/other.yml@refs/pull/1/merge'}), /entry workflow/);
});

test('only direct protected-main hardware dispatches are exempt from PR reviews', async () => {
  const caller = repo + '/.github/workflows/omni-ci.yaml@refs/heads/main';
  await check({event: 'workflow_dispatch', caller});
  await assert.rejects(check({event: 'workflow_dispatch', caller, ref: 'refs/heads/dev'}), /protected main/);
  await assert.rejects(check({event: 'workflow_dispatch', caller: caller.replace('omni-ci', 'other')}), /protected main/);
  await assert.rejects(check({event: 'issue_comment', caller}), /pull request event/);
});
