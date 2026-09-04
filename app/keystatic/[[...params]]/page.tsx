'use client';

export const dynamic = 'force-static';

import { makePage } from '@keystatic/next/ui/app';
import keystaticConfig from '../../../keystatic.config';

export default makePage(keystaticConfig);
