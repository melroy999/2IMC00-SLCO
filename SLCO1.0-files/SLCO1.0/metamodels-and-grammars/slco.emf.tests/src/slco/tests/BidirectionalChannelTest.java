/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.BidirectionalChannel;
import slco.SlcoFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Bidirectional Channel</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class BidirectionalChannelTest extends ChannelTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(BidirectionalChannelTest.class);
	}

	/**
	 * Constructs a new Bidirectional Channel test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BidirectionalChannelTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Bidirectional Channel test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected BidirectionalChannel getFixture() {
		return (BidirectionalChannel)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createBidirectionalChannel());
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#tearDown()
	 * @generated
	 */
	@Override
	protected void tearDown() throws Exception {
		setFixture(null);
	}

} //BidirectionalChannelTest
