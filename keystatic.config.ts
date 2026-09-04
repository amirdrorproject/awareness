import { config, fields, singleton } from '@keystatic/core';

export default config({
  storage: {
    kind: 'github',
    repo: 'amirdrorproject/awareness',
    branchPrefix: 'langgraph',
  },
  singletons: {
    messages: singleton({
      label: 'messages',
      path: 'langgraph_poc/messages',
      format: 'json',
      schema: {
        _shared: fields.object({
          AGENT_DISCLOSURE_TEXT: fields.text({
            label: 'AGENT_DISCLOSURE_TEXT',
            multiline: true,
          }),
        }),
        respond_direct: fields.text({
          label: 'respond_direct',
          multiline: true,
        }),
        respond_with_check: fields.object({
          situation_ack: fields.text({
            label: 'situation_ack',
            multiline: true,
          }),
          dilemma_ack: fields.text({
            label: 'dilemma_ack',
            multiline: true,
          }),
          suffix: fields.text({
            label: 'suffix',
            multiline: true,
          }),
        }),
        present_success_analysis_intro: fields.text({
          label: 'present_success_analysis_intro',
          multiline: true,
        }),
        invite_success_story: fields.text({
          label: 'invite_success_story',
          multiline: true,
        }),
        pivot_to_deeper_process: fields.text({
          label: 'pivot_to_deeper_process',
          multiline: true,
        }),
        present_practical_track_intro: fields.text({
          label: 'present_practical_track_intro',
          multiline: true,
        }),
        pivot_practical_to_success: fields.text({
          label: 'pivot_practical_to_success',
          multiline: true,
        }),
      },
    }),
    prompts: singleton({
      label: 'prompts',
      path: 'langgraph_poc/prompts',
      format: 'json',
      schema: {
        success_analysis_conversation: fields.text({
          label: 'success_analysis_conversation',
          multiline: true,
        }),
        practical_track_conversation: fields.text({
          label: 'practical_track_conversation',
          multiline: true,
        }),
        ask_direction: fields.object({
          emotional_vague: fields.text({
            label: 'emotional_vague',
            multiline: true,
          }),
          dual: fields.text({
            label: 'dual',
            multiline: true,
          }),
        }),
        invite_to_share: fields.text({
          label: 'invite_to_share',
          multiline: true,
        }),
        present_and_ask: fields.text({
          label: 'present_and_ask',
          multiline: true,
        }),
        ask_which_block: fields.text({
          label: 'ask_which_block',
          multiline: true,
        }),
        deepen_reply: fields.text({
          label: 'deepen_reply',
          multiline: true,
        }),
        focus_on_block: fields.text({
          label: 'focus_on_block',
          multiline: true,
        }),
        summarize_and_pivot: fields.text({
          label: 'summarize_and_pivot',
          multiline: true,
        }),
        explain_success_value: fields.text({
          label: 'explain_success_value',
          multiline: true,
        }),
      },
    }),
  },
});
