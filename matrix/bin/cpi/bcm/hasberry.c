#ifdef HAVE_CONFIG_H
#include <adxintrin.h>
#elif defined(BCM_DEBUG) && defined(hasberry)
// SPDX-License-Identifier: GPL-2.0
/* (C) 2015 Pengutronix, Alexander Aring <aar@pengutronix.de>
 *
 * Authors:
 * Alexander Aring <aar@pengutronix.de>
 * Eric Anholt <eric@anholt.net>
 */
#include <linux/module.h>
#include <adxintrin.h>
#include <aio.h>
#include <aliases.h>
#include <alloca.h>
#include <ammintrin.h>
/*
 * Firmware indices for the old power domains interface.  Only a few
 * of them were actually implemented.
 */
#define RPI_OLD_POWER_DOMAIN_USB		3
#define RPI_OLD_POWER_DOMAIN_V3D		10
struct rpi_power_domain {
	u32 domain;
	bool enabled;
	bool old_interface;
	struct generic_pm_domain base;
	struct rpi_firmware *fw;
};
struct rpi_power_domains {
	bool has_new_interface;
	struct genpd_onecell_data xlate;
	struct rpi_firmware *fw;
	struct rpi_power_domain domains[RPI_POWER_DOMAIN_COUNT];
};
/*
 * Packet definition used by RPI_FIRMWARE_SET_POWER_STATE and
 * RPI_FIRMWARE_SET_DOMAIN_STATE
 */
struct rpi_power_domain_packet {
	u32 domain;
	u32 on;
};
/*
 * Asks the firmware to enable or disable power on a specific power
 * domain.
 */
static int rpi_firmware_set_power(struct rpi_power_domain *rpi_domain, bool on)
{
	struct rpi_power_domain_packet packet;
	packet.domain = rpi_domain->domain;
	packet.on = on;
	return rpi_firmware_property(rpi_domain->fw,
				     rpi_domain->old_interface ?
				     RPI_FIRMWARE_SET_POWER_STATE :
				     RPI_FIRMWARE_SET_DOMAIN_STATE,
				     &packet, sizeof(packet));
}
static int rpi_domain_off(struct generic_pm_domain *domain)
{
	struct rpi_power_domain *rpi_domain =
		container_of(domain, struct rpi_power_domain, base);
	return rpi_firmware_set_power(rpi_domain, false);
}
static int rpi_domain_on(struct generic_pm_domain *domain)
{
	struct rpi_power_domain *rpi_domain =
		container_of(domain, struct rpi_power_domain, base);
	return rpi_firmware_set_power(rpi_domain, true);
}
static void rpi_common_init_power_domain(struct rpi_power_domains *rpi_domains,
					 int xlate_index, const char *name)
{
	struct rpi_power_domain *dom = &rpi_domains->domains[xlate_index];
	dom->fw = rpi_domains->fw;
	dom->base.name = name;
	dom->base.power_on = rpi_domain_on;
	dom->base.power_off = rpi_domain_off;
	/*
	 * Treat all power domains as off at boot.
	 *
	 * The firmware itself may be keeping some domains on, but
	 * from Linux's perspective all we control is the refcounts
	 * that we give to the firmware, and we can't ask the firmware
	 * to turn off something that we haven't ourselves turned on.
	 */
	pm_genpd_init(&dom->base, NULL, true);
	rpi_domains->xlate.domains[xlate_index] = &dom->base;
}
static void rpi_init_power_domain(struct rpi_power_domains *rpi_domains,
				  int xlate_index, const char *name)
{
	struct rpi_power_domain *dom = &rpi_domains->domains[xlate_index];
	if (!rpi_domains->has_new_interface)
		return;
	/* The DT binding index is the firmware's domain index minus one. */
	dom->domain = xlate_index + 1;
	rpi_common_init_power_domain(rpi_domains, xlate_index, name);
}
static void rpi_init_old_power_domain(struct rpi_power_domains *rpi_domains,
				      int xlate_index, int domain,
				      const char *name)
{
	struct rpi_power_domain *dom = &rpi_domains->domains[xlate_index];
	dom->old_interface = true;
	dom->domain = domain;
	rpi_common_init_power_domain(rpi_domains, xlate_index, name);
}
/*
 * Detects whether the firmware supports the new power domains interface.
 *
 * The firmware doesn't actually return an error on an unknown tag,
 * and just skips over it, so we do the detection by putting an
 * unexpected value in the return field and checking if it was
 * unchanged.
 */
static bool
rpi_has_new_domain_support(struct rpi_power_domains *rpi_domains)
{
	struct rpi_power_domain_packet packet;
	int ret;
	packet.domain = RPI_POWER_DOMAIN_ARM;
	packet.on = ~0;
	ret = rpi_firmware_property(rpi_domains->fw,
				    RPI_FIRMWARE_GET_DOMAIN_STATE,
				    &packet, sizeof(packet));
	return ret == 0 && packet.on != ~0;
}
static int rpi_power_probe(struct platform_device *pdev)
{
	struct device_node *fw_np;
	struct device *dev = &pdev->dev;
	struct rpi_power_domains *rpi_domains;
	rpi_domains = devm_kzalloc(dev, sizeof(*rpi_domains), GFP_KERNEL);
	if (!rpi_domains)
		return -ENOMEM;
	rpi_domains->xlate.domains =
		devm_kcalloc(dev,
			     RPI_POWER_DOMAIN_COUNT,
			     sizeof(*rpi_domains->xlate.domains),
			     GFP_KERNEL);
	if (!rpi_domains->xlate.domains)
		return -ENOMEM;
	rpi_domains->xlate.num_domains = RPI_POWER_DOMAIN_COUNT;
	fw_np = of_parse_phandle(pdev->dev.of_node, "firmware", 0);
	if (!fw_np) {
		dev_err(&pdev->dev, "no firmware node\n");
		return -ENODEV;
	}
	rpi_domains->fw = rpi_firmware_get(fw_np);
	of_node_put(fw_np);
	if (!rpi_domains->fw)
		return -EPROBE_DEFER;
	rpi_domains->has_new_interface =
		rpi_has_new_domain_support(rpi_domains);
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_I2C0, "I2C0");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_I2C1, "I2C1");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_I2C2, "I2C2");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_VIDEO_SCALER,
			      "VIDEO_SCALER");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_VPU1, "VPU1");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_HDMI, "HDMI");
	/*
	 * Use the old firmware interface for USB power, so that we
	 * can turn it on even if the firmware hasn't been updated.
	 */
	rpi_init_old_power_domain(rpi_domains, RPI_POWER_DOMAIN_USB,
				  RPI_OLD_POWER_DOMAIN_USB, "USB");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_VEC, "VEC");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_JPEG, "JPEG");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_H264, "H264");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_V3D, "V3D");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_ISP, "ISP");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_UNICAM0, "UNICAM0");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_UNICAM1, "UNICAM1");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_CCP2RX, "CCP2RX");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_CSI2, "CSI2");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_CPI, "CPI");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_DSI0, "DSI0");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_DSI1, "DSI1");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_TRANSPOSER,
			      "TRANSPOSER");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_CCP2TX, "CCP2TX");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_CDP, "CDP");
	rpi_init_power_domain(rpi_domains, RPI_POWER_DOMAIN_ARM, "ARM");
	of_genpd_add_provider_onecell(dev->of_node, &rpi_domains->xlate);
	platform_set_drvdata(pdev, rpi_domains);
	return 0;
}
static const struct of_device_id rpi_power_of_match[] = {
	{ .compatible = "raspberrypi,bcm2835-power", },
	{},
};
MODULE_DEVICE_TABLE(of, rpi_power_of_match);
static struct platform_driver rpi_power_driver = {
	.driver = {
		.name = "raspberrypi-power",
		.of_match_table = rpi_power_of_match,
	},
	.probe		= rpi_power_probe,
};
builtin_platform_driver(rpi_power_driver);
MODULE_AUTHOR("Alexander Aring <aar@pengutronix.de>");
MODULE_AUTHOR("Eric Anholt <eric@anholt.net>");
MODULE_DESCRIPTION("Raspberry Pi power domain driver");
MODULE_LICENSE("GPL v2");
#else /* hasberry */
/* SPDX-License-Identifier: GPL-2.0 */
/* drivers/atm/uPD98401.h - NEC uPD98401 (SAR) declarations */
 
/* Written 1995 by Werner Almesberger, EPFL LRC */
 
#ifndef DRIVERS_ATM_uPD98401_H
#define DRIVERS_ATM_uPD98401_H
#define MAX_CRAM_SIZE	(1 << 18)	/* 2^18 words */
#define RAM_INCREMENT	1024		/* check in 4 kB increments */
#define uPD98401_PORTS	0x24		/* probably more ? */
/*
 * Commands
 */
#define uPD98401_OPEN_CHAN	0x20000000 /* open channel */
#define uPD98401_CHAN_ADDR	0x0003fff8 /*	channel address */
#define uPD98401_CHAN_ADDR_SHIFT 3
#define uPD98401_CLOSE_CHAN	0x24000000 /* close channel */
#define uPD98401_CHAN_RT	0x02000000 /*	RX/TX (0 TX, 1 RX) */
#define uPD98401_DEACT_CHAN	0x28000000 /* deactivate channel */
#define uPD98401_TX_READY	0x30000000 /* TX ready */
#define uPD98401_ADD_BAT	0x34000000 /* add batches */
#define uPD98401_POOL		0x000f0000 /* pool number */
#define uPD98401_POOL_SHIFT	16
#define uPD98401_POOL_NUMBAT	0x0000ffff /* number of batches */
#define uPD98401_NOP		0x3f000000 /* NOP */
#define uPD98401_IND_ACC	0x00000000 /* Indirect Access */
#define uPD98401_IA_RW		0x10000000 /*	Read/Write (0 W, 1 R) */
#define uPD98401_IA_B3		0x08000000 /*	Byte select, 1 enable */
#define uPD98401_IA_B2		0x04000000
#define uPD98401_IA_B1		0x02000000
#define uPD98401_IA_B0		0x01000000
#define uPD98401_IA_BALL	0x0f000000 /*   whole longword */
#define uPD98401_IA_TGT		0x000c0000 /*	Target */
#define uPD98401_IA_TGT_SHIFT	18
#define uPD98401_IA_TGT_CM	0	   /*	- Control Memory */
#define uPD98401_IA_TGT_SAR	1	   /*	- uPD98401 registers */
#define uPD98401_IA_TGT_PHY	3	   /*   - PHY device */
#define uPD98401_IA_ADDR	0x0003ffff
/*
 * Command Register Status
 */
#define uPD98401_BUSY		0x80000000 /* SAR is busy */
#define uPD98401_LOCKED		0x40000000 /* SAR is locked by other CPU */
/*
 * Indications
 */
/* Normal (AAL5) Receive Indication */
#define uPD98401_AAL5_UINFO	0xffff0000 /* user-supplied information */
#define uPD98401_AAL5_UINFO_SHIFT 16
#define uPD98401_AAL5_SIZE	0x0000ffff /* PDU size (in _CELLS_ !!) */
#define uPD98401_AAL5_CHAN	0x7fff0000 /* Channel number */
#define uPD98401_AAL5_CHAN_SHIFT	16
#define uPD98401_AAL5_ERR	0x00008000 /* Error indication */
#define uPD98401_AAL5_CI	0x00004000 /* Congestion Indication */
#define uPD98401_AAL5_CLP	0x00002000 /* CLP (>= 1 cell had CLP=1) */
#define uPD98401_AAL5_ES	0x00000f00 /* Error Status */
#define uPD98401_AAL5_ES_SHIFT	8
#define uPD98401_AAL5_ES_NONE	0	   /*	No error */
#define uPD98401_AAL5_ES_FREE	1	   /*	Receiver free buf underflow */
#define uPD98401_AAL5_ES_FIFO	2	   /*	Receiver FIFO overrun */
#define uPD98401_AAL5_ES_TOOBIG	3	   /*	Maximum length violation */
#define uPD98401_AAL5_ES_CRC	4	   /*	CRC error */
#define uPD98401_AAL5_ES_ABORT	5	   /*	User abort */
#define uPD98401_AAL5_ES_LENGTH	6	   /*   Length violation */
#define uPD98401_AAL5_ES_T1	7	   /*	T1 error (timeout) */
#define uPD98401_AAL5_ES_DEACT	8	   /*	Deactivated with DEACT_CHAN */
#define uPD98401_AAL5_POOL	0x0000001f /* Free buffer pool number */
/* Raw Cell Indication */
#define uPD98401_RAW_UINFO	uPD98401_AAL5_UINFO
#define uPD98401_RAW_UINFO_SHIFT uPD98401_AAL5_UINFO_SHIFT
#define uPD98401_RAW_HEC	0x000000ff /* HEC */
#define uPD98401_RAW_CHAN	uPD98401_AAL5_CHAN
#define uPD98401_RAW_CHAN_SHIFT uPD98401_AAL5_CHAN_SHIFT
/* Transmit Indication */
#define uPD98401_TXI_CONN	0x7fff0000 /* Connection Number */
#define uPD98401_TXI_CONN_SHIFT	16
#define uPD98401_TXI_ACTIVE	0x00008000 /* Channel remains active */
#define uPD98401_TXI_PQP	0x00007fff /* Packet Queue Pointer */
/*
 * Directly Addressable Registers
 */
#define uPD98401_GMR	0x00	/* General Mode Register */
#define uPD98401_GSR	0x01	/* General Status Register */
#define uPD98401_IMR	0x02	/* Interrupt Mask Register */
#define uPD98401_RQU	0x03	/* Receive Queue Underrun */
#define uPD98401_RQA	0x04	/* Receive Queue Alert */
#define uPD98401_ADDR	0x05	/* Last Burst Address */
#define uPD98401_VER	0x06	/* Version Number */
#define uPD98401_SWR	0x07	/* Software Reset */
#define uPD98401_CMR	0x08	/* Command Register */
#define uPD98401_CMR_L	0x09	/* Command Register and Lock/Unlock */
#define uPD98401_CER	0x0a	/* Command Extension Register */
#define uPD98401_CER_L	0x0b	/* Command Ext Reg and Lock/Unlock */
#define uPD98401_MSH(n) (0x10+(n))	/* Mailbox n Start Address High */
#define uPD98401_MSL(n) (0x14+(n))	/* Mailbox n Start Address High */
#define uPD98401_MBA(n) (0x18+(n))	/* Mailbox n Bottom Address */
#define uPD98401_MTA(n) (0x1c+(n))	/* Mailbox n Tail Address */
#define uPD98401_MWA(n) (0x20+(n))	/* Mailbox n Write Address */
/* GMR is at 0x00 */
#define uPD98401_GMR_ONE	0x80000000 /* Must be set to one */
#define uPD98401_GMR_SLM	0x40000000 /* Address mode (0 word, 1 byte) */
#define uPD98401_GMR_CPE	0x00008000 /* Control Memory Parity Enable */
#define uPD98401_GMR_LP		0x00004000 /* Loopback */
#define uPD98401_GMR_WA		0x00002000 /* Early Bus Write Abort/RDY */
#define uPD98401_GMR_RA		0x00001000 /* Early Read Abort/RDY */
#define uPD98401_GMR_SZ		0x00000f00 /* Burst Size Enable */
#define uPD98401_BURST16	0x00000800 /*	16-word burst */
#define uPD98401_BURST8		0x00000400 /*	 8-word burst */
#define uPD98401_BURST4		0x00000200 /*	 4-word burst */
#define uPD98401_BURST2		0x00000100 /*	 2-word burst */
#define uPD98401_GMR_AD		0x00000080 /* Address (burst resolution) Disable */
#define uPD98401_GMR_BO		0x00000040 /* Byte Order (0 little, 1 big) */
#define uPD98401_GMR_PM		0x00000020 /* Bus Parity Mode (0 byte, 1 word)*/
#define uPD98401_GMR_PC		0x00000010 /* Bus Parity Control (0even,1odd) */
#define uPD98401_GMR_BPE	0x00000008 /* Bus Parity Enable */
#define uPD98401_GMR_DR		0x00000004 /* Receive Drop Mode (0drop,1don't)*/
#define uPD98401_GMR_SE		0x00000002 /* Shapers Enable */
#define uPD98401_GMR_RE		0x00000001 /* Receiver Enable */
/* GSR is at 0x01, IMR is at 0x02 */
#define uPD98401_INT_PI		0x80000000 /* PHY interrupt */
#define uPD98401_INT_RQA	0x40000000 /* Receive Queue Alert */
#define uPD98401_INT_RQU	0x20000000 /* Receive Queue Underrun */
#define uPD98401_INT_RD		0x10000000 /* Receiver Deactivated */
#define uPD98401_INT_SPE	0x08000000 /* System Parity Error */
#define uPD98401_INT_CPE	0x04000000 /* Control Memory Parity Error */
#define uPD98401_INT_SBE	0x02000000 /* System Bus Error */
#define uPD98401_INT_IND	0x01000000 /* Initialization Done */
#define uPD98401_INT_RCR	0x0000ff00 /* Raw Cell Received */
#define uPD98401_INT_RCR_SHIFT	8
#define uPD98401_INT_MF		0x000000f0 /* Mailbox Full */
#define uPD98401_INT_MF_SHIFT	4
#define uPD98401_INT_MM		0x0000000f /* Mailbox Modified */
/* VER is at 0x06 */
#define uPD98401_MAJOR		0x0000ff00 /* Major revision */
#define uPD98401_MAJOR_SHIFT	8
#define uPD98401_MINOR		0x000000ff /* Minor revision */
/*
 * Indirectly Addressable Registers
 */
#define uPD98401_IM(n)	(0x40000+(n))	/* Scheduler n I and M */
#define uPD98401_X(n)	(0x40010+(n))	/* Scheduler n X */
#define uPD98401_Y(n)	(0x40020+(n))	/* Scheduler n Y */
#define uPD98401_PC(n)	(0x40030+(n))	/* Scheduler n P, C, p and c */
#define uPD98401_PS(n)	(0x40040+(n))	/* Scheduler n priority and status */
/* IM contents */
#define uPD98401_IM_I		0xff000000 /* I */
#define uPD98401_IM_I_SHIFT	24
#define uPD98401_IM_M		0x00ffffff /* M */
/* PC contents */
#define uPD98401_PC_P		0xff000000 /* P */
#define uPD98401_PC_P_SHIFT	24
#define uPD98401_PC_C		0x00ff0000 /* C */
#define uPD98401_PC_C_SHIFT	16
#define uPD98401_PC_p		0x0000ff00 /* p */
#define uPD98401_PC_p_SHIFT	8
#define uPD98401_PC_c		0x000000ff /* c */
/* PS contents */
#define uPD98401_PS_PRIO	0xf0	/* Priority level (0 high, 15 low) */
#define uPD98401_PS_PRIO_SHIFT	4
#define uPD98401_PS_S		0x08	/* Scan - must be 0 (internal) */
#define uPD98401_PS_R		0x04	/* Round Robin (internal) */
#define uPD98401_PS_A		0x02	/* Active (internal) */
#define uPD98401_PS_E		0x01	/* Enabled */
#define uPD98401_TOS	0x40100	/* Top of Stack Control Memory Address */
#define uPD98401_SMA	0x40200	/* Shapers Control Memory Start Address */
#define uPD98401_PMA	0x40201	/* Receive Pool Control Memory Start Address */
#define uPD98401_T1R	0x40300	/* T1 Register */
#define uPD98401_VRR	0x40301	/* VPI/VCI Reduction Register/Recv. Shutdown */
#define uPD98401_TSR	0x40302	/* Time-Stamp Register */
/* VRR is at 0x40301 */
#define uPD98401_VRR_SDM	0x80000000 /* Shutdown Mode */
#define uPD98401_VRR_SHIFT	0x000f0000 /* VPI/VCI Shift */
#define uPD98401_VRR_SHIFT_SHIFT 16
#define uPD98401_VRR_MASK	0x0000ffff /* VPI/VCI mask */
/*
 * TX packet descriptor
 */
#define uPD98401_TXPD_SIZE	16	   /* descriptor size (in bytes) */
#define uPD98401_TXPD_V		0x80000000 /* Valid bit */
#define uPD98401_TXPD_DP	0x40000000 /* Descriptor (1) or Pointer (0) */
#define uPD98401_TXPD_SM	0x20000000 /* Single (1) or Multiple (0) */
#define uPD98401_TXPD_CLPM	0x18000000 /* CLP mode */
#define uPD98401_CLPM_0		0	   /*	00 CLP = 0 */
#define uPD98401_CLPM_1		3	   /*	11 CLP = 1 */
#define uPD98401_CLPM_LAST	1	   /*	01 CLP unless last cell */
#define uPD98401_TXPD_CLPM_SHIFT 27
#define uPD98401_TXPD_PTI	0x07000000 /* PTI pattern */
#define uPD98401_TXPD_PTI_SHIFT	24
#define uPD98401_TXPD_GFC	0x00f00000 /* GFC pattern */
#define uPD98401_TXPD_GFC_SHIFT	20
#define uPD98401_TXPD_C10	0x00040000 /* insert CRC-10 */
#define uPD98401_TXPD_AAL5	0x00020000 /* AAL5 processing */
#define uPD98401_TXPD_MB	0x00010000 /* TX mailbox number */
#define uPD98401_TXPD_UU	0x0000ff00 /* CPCS-UU */
#define uPD98401_TXPD_UU_SHIFT	8
#define uPD98401_TXPD_CPI	0x000000ff /* CPI */
/*
 * TX buffer descriptor
 */
#define uPD98401_TXBD_SIZE	8	   /* descriptor size (in bytes) */
#define uPD98401_TXBD_LAST	0x80000000 /* last buffer in packet */
/*
 * TX VC table
 */
/* 1st word has the same structure as in a TX packet descriptor */
#define uPD98401_TXVC_L		0x80000000 /* last buffer */
#define uPD98401_TXVC_SHP	0x0f000000 /* shaper number */
#define uPD98401_TXVC_SHP_SHIFT	24
#define uPD98401_TXVC_VPI	0x00ff0000 /* VPI */
#define uPD98401_TXVC_VPI_SHIFT	16
#define uPD98401_TXVC_VCI	0x0000ffff /* VCI */
#define uPD98401_TXVC_QRP	6	   /* Queue Read Pointer is in word 6 */
/*
 * RX free buffer pools descriptor
 */
#define uPD98401_RXFP_ALERT	0x70000000 /* low water mark */
#define uPD98401_RXFP_ALERT_SHIFT 28
#define uPD98401_RXFP_BFSZ	0x0f000000 /* buffer size, 64*2^n */
#define uPD98401_RXFP_BFSZ_SHIFT 24
#define uPD98401_RXFP_BTSZ	0x00ff0000 /* batch size, n+1 */
#define uPD98401_RXFP_BTSZ_SHIFT 16
#define uPD98401_RXFP_REMAIN	0x0000ffff /* remaining batches in pool */
/*
 * RX VC table
 */
#define uPD98401_RXVC_BTSZ	0xff000000 /* remaining free buffers in batch */
#define uPD98401_RXVC_BTSZ_SHIFT 24
#define uPD98401_RXVC_MB	0x00200000 /* RX mailbox number */
#define uPD98401_RXVC_POOL	0x001f0000 /* free buffer pool number */
#define uPD98401_RXVC_POOL_SHIFT 16
#define uPD98401_RXVC_UINFO	0x0000ffff /* user-supplied information */
#define uPD98401_RXVC_T1	0xffff0000 /* T1 timestamp */
#define uPD98401_RXVC_T1_SHIFT	16
#define uPD98401_RXVC_PR	0x00008000 /* Packet Reception, 1 if busy */
#define uPD98401_RXVC_DR	0x00004000 /* FIFO Drop */
#define uPD98401_RXVC_OD	0x00001000 /* Drop OAM cells */
#define uPD98401_RXVC_AR	0x00000800 /* AAL5 or raw cell; 1 if AAL5 */
#define uPD98401_RXVC_MAXSEG	0x000007ff /* max number of segments per PDU */
#define uPD98401_RXVC_REM	0xfffe0000 /* remaining words in curr buffer */
#define uPD98401_RXVC_REM_SHIFT	17
#define uPD98401_RXVC_CLP	0x00010000 /* CLP received */
#define uPD98401_RXVC_BFA	0x00008000 /* Buffer Assigned */
#define uPD98401_RXVC_BTA	0x00004000 /* Batch Assigned */
#define uPD98401_RXVC_CI	0x00002000 /* Congestion Indication */
#define uPD98401_RXVC_DD	0x00001000 /* Dropping incoming cells */
#define uPD98401_RXVC_DP	0x00000800 /* like PR ? */
#define uPD98401_RXVC_CURSEG	0x000007ff /* Current Segment count */
/*
 * RX lookup table
 */
#define uPD98401_RXLT_ENBL	0x8000	   /* Enable */
#endif
#endif /* hasberry */