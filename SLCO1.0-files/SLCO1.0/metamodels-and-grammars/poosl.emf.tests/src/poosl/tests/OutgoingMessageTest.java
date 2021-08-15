/**
 */
package poosl.tests;

import junit.textui.TestRunner;

import poosl.OutgoingMessage;
import poosl.PooslFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Outgoing Message</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class OutgoingMessageTest extends MessageTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(OutgoingMessageTest.class);
	}

	/**
	 * Constructs a new Outgoing Message test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutgoingMessageTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Outgoing Message test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected OutgoingMessage getFixture() {
		return (OutgoingMessage)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PooslFactory.eINSTANCE.createOutgoingMessage());
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

} //OutgoingMessageTest
