import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics',
      collapsed: false,
      items: [
        'intro',
        {
          type: 'category',
          label: 'Part I: Foundations of Physical AI',
          collapsed: false,
          items: [
            'physical-ai/fundamentals/math-fundamentals',
            'physical-ai/fundamentals/sensors-perception'
          ]
        },
        {
          type: 'category',
          label: 'Part II: Kinematics and Dynamics',
          collapsed: false,
          items: [
            'physical-ai/kinematics/forward-kinematics',
            'physical-ai/kinematics/inverse-kinematics',
            'physical-ai/locomotion/gait-planning'
          ]
        },
        {
          type: 'category',
          label: 'Part III: Control and Learning',
          collapsed: false,
          items: [
            'physical-ai/learning/ml-for-robotics',
            'physical-ai/learning/reinforcement-learning'
          ]
        },
        {
          type: 'category',
          label: 'Part IV: Applications and Integration',
          collapsed: false,
          items: [
            'physical-ai/interaction/human-robot-interaction',
            'physical-ai/applications/safety-ethics'
          ]
        }
      ]
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: true,
      items: [
        'reference/glossary',
        'reference/abbreviations',
        'reference/further-reading'
      ]
    },
    {
      type: 'category',
      label: 'Tutorials',
      collapsed: true,
      items: [
        'tutorials/setup-environment',
        'tutorials/simulation-basics',
        'tutorials/code-examples'
      ]
    }
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
